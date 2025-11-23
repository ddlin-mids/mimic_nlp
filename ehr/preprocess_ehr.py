import argparse
import copy
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data.data_utils import DEMO_COLS, LAB_COLS  # noqa: E402
from data.data_utils.readmission_utils import get_readmission_label_mimic  # noqa: E402

COLS_IRRELEVANT = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "split",
    "splits",
    "date",
    "node_name",
]
CAT_COLUMNS = LAB_COLS + ["gender", "race"]

SUBGOUPRS_EXCLUDED = [
    "Z00-Z13",
    "Z14-Z15",
    "Z16-Z16",
    "Z17-Z17",
    "Z18-Z18",
    "Z19-Z19",
    "Z20-Z29",
    "Z30-Z39",
    "Z40-Z53",
    "Z55-Z65",
    "Z66-Z66",
    "Z67-Z67",
    "Z68-Z68",
    "Z69-Z76",
    "Z77-Z99",
]

ICD_MAP_PATH = REPO_ROOT / "refs/readmit-stgnn/data/ICD10_Groups.csv"
NDC_MAP_PATH = REPO_ROOT / "refs/readmit-stgnn/data/ndc2therapeutic.csv"


def load_cohort(path: str) -> pd.DataFrame:
    """Load cohort CSV with expected date parsing."""
    return pd.read_csv(path, parse_dates=["admittime", "dischtime"])


def preprocess_icd(path: str, df_demo: pd.DataFrame) -> pd.DataFrame:
    """
    Load ICD diagnoses for the cohort admissions.
    Minimal mapping: strip dots, uppercase, and take 3-char prefix as SUBGROUP.
    """
    hadm_set = set(df_demo["hadm_id"])
    frames = []
    for chunk in pd.read_csv(path, chunksize=500_000, usecols=["hadm_id", "icd_code"]):
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        if chunk.empty:
            continue
        icd = chunk["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
        chunk = chunk.assign(SUBGROUP=icd.str[:3])
        frames.append(chunk[["hadm_id", "SUBGROUP"]])
    if not frames:
        return pd.DataFrame(columns=["hadm_id", "SUBGROUP"])
    return pd.concat(frames, ignore_index=True)


def preprocess_lab(path: str, df_demo: pd.DataFrame) -> pd.DataFrame:
    """
    Load labs for the cohort admissions and derive Day_Number and abnormal flags.

    - Filters to cohort hadm_ids.
    - Computes Day_Number relative to admittime.
    - Keeps rows within [1, LOS].
    - Marks abnormal if any flag == "abnormal" for that hadm/day/lab.
    - Returns a melted dataframe ready for pivoting in the per-day aggregation.
    """
    hadm_set = set(df_demo["hadm_id"])
    adm_map = df_demo.set_index("hadm_id")["admittime"].to_dict()
    dis_map = df_demo.set_index("hadm_id")["dischtime"].to_dict()
    los_map = df_demo.set_index("hadm_id")["length_of_stay_days"].to_dict()

    frames = []
    for chunk in pd.read_csv(
        path,
        chunksize=1_000_000,
        usecols=["hadm_id", "itemid", "charttime", "flag"],
        dtype={"flag": str},
    ):
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["admittime"] = chunk["hadm_id"].map(adm_map)
        chunk["dischtime"] = chunk["hadm_id"].map(dis_map)
        chunk = chunk.dropna(subset=["charttime", "admittime"])
        chunk["day_num"] = (chunk["charttime"].dt.floor("D") - chunk["admittime"].dt.floor("D")).dt.days + 1
        # keep within LOS window
        chunk["los"] = chunk["hadm_id"].map(los_map)
        chunk = chunk[(chunk["day_num"] >= 1) & (chunk["day_num"] <= chunk["los"])]
        if chunk.empty:
            continue
        chunk = chunk.assign(
            label_fluid=chunk["itemid"].astype(str),
            flag=chunk["flag"].fillna("nan").str.lower(),
            Day_Number=chunk["day_num"].astype(float),
        )
        frames.append(chunk[["hadm_id", "label_fluid", "flag", "Day_Number"]])
    if not frames:
        return pd.DataFrame(columns=["hadm_id", "label_fluid", "flag", "Day_Number"])
    labs = pd.concat(frames, ignore_index=True)

    # Compute abnormal per hadm/day/lab
    labs["is_abnormal"] = labs["flag"].eq("abnormal")
    grouped = labs.groupby(["hadm_id", "Day_Number", "label_fluid"])["is_abnormal"].any().reset_index()
    grouped["flag"] = grouped["is_abnormal"].map({True: "abnormal", False: "nan"})
    grouped = grouped.drop(columns=["is_abnormal"])
    return grouped


def preprocess_med(path: str, df_demo: pd.DataFrame) -> pd.DataFrame:
    """
    Load meds for the cohort admissions and derive Day_Number + therapeutic class.
    """
    hadm_set = set(df_demo["hadm_id"])
    adm_map = df_demo.set_index("hadm_id")["admittime"].to_dict()
    dis_map = df_demo.set_index("hadm_id")["dischtime"].to_dict()
    los_map = df_demo.set_index("hadm_id")["length_of_stay_days"].to_dict()

    # load ndc -> therapeutic class map if available
    ndc_map = {}
    if NDC_MAP_PATH.exists():
        ndc_df = pd.read_csv(NDC_MAP_PATH)
        ndc_map = dict(zip(ndc_df["NDC_MEDICATION_CODE"].astype(str), ndc_df["MED_THERAPEUTIC_CLASS_DESCRIPTION"]))

    frames = []
    usecols = ["hadm_id", "ndc", "drug", "starttime", "stoptime"]
    for chunk in pd.read_csv(path, chunksize=500_000, usecols=usecols):
        chunk = chunk[chunk["hadm_id"].isin(hadm_set)]
        if chunk.empty:
            continue
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["admittime"] = chunk["hadm_id"].map(adm_map)
        chunk["dischtime"] = chunk["hadm_id"].map(dis_map)
        chunk = chunk.dropna(subset=["admittime"])
        # compute day number from starttime (fallback to admittime if missing)
        start = chunk["starttime"].fillna(chunk["admittime"])
        chunk["day_num"] = (start.dt.floor("D") - chunk["admittime"].dt.floor("D")).dt.days + 1
        chunk["los"] = chunk["hadm_id"].map(los_map)
        chunk = chunk[(chunk["day_num"] >= 1) & (chunk["day_num"] <= chunk["los"])]
        if chunk.empty:
            continue
        # map ndc to therapeutic class; fallback to drug name
        chunk["ndc_str"] = chunk["ndc"].astype(str)
        chunk["MED_THERAPEUTIC_CLASS_DESCRIPTION"] = chunk["ndc_str"].map(ndc_map)
        chunk["MED_THERAPEUTIC_CLASS_DESCRIPTION"] = chunk["MED_THERAPEUTIC_CLASS_DESCRIPTION"].fillna(
            chunk["drug"].astype(str).str.upper()
        )
        chunk = chunk.assign(Day_Number=chunk["day_num"].astype(float))
        frames.append(chunk[["hadm_id", "MED_THERAPEUTIC_CLASS_DESCRIPTION", "Day_Number"]])

    if not frames:
        return pd.DataFrame(columns=["hadm_id", "MED_THERAPEUTIC_CLASS_DESCRIPTION", "Day_Number"])
    return pd.concat(frames, ignore_index=True)


def ehr_bag_of_words_mimic(
    df_demo, df_ehr, col_name, time_step_by="day", filter_freq=None
):
    """
    Vectorized EHR bag-of-words processing.

    Creates daily counts of features (ICD codes, medications) per admission.
    Uses pandas vectorized operations instead of nested loops for 100-1000x speedup.

    Args:
        df_demo: demographics dataframe with subject_id, hadm_id, admittime, dischtime, etc.
        df_ehr: EHR events dataframe (ICD or meds) with hadm_id, Day_Number, and feature column
        col_name: column name containing the feature values (e.g., 'SUBGROUP', 'MED_THERAPEUTIC_CLASS_DESCRIPTION')
        time_step_by: only 'day' is supported
        filter_freq: minimum frequency threshold to keep a feature (optional)

    Returns:
        DataFrame with one row per admission-day, with feature counts as columns
    """
    import warnings
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    # Filter valid feature values once
    all_values = df_ehr[col_name].unique()
    all_values = [
        val for val in all_values
        if isinstance(val, str) and (val not in SUBGOUPRS_EXCLUDED)
    ]
    print(f"Processing {len(all_values)} unique feature values...")

    # Prepare demographics: add admit/discharge dates and create date ranges
    df_demo = df_demo.copy()
    df_demo['admit_date'] = pd.to_datetime(df_demo['admittime']).dt.date
    df_demo['discharge_date'] = pd.to_datetime(df_demo['dischtime']).dt.date

    # Create expanded dataframe: one row per admission-day
    # This is the key vectorization: use explode instead of nested loops
    df_expanded = df_demo.copy()
    df_expanded['date_range'] = df_expanded.apply(
        lambda row: pd.date_range(row['admit_date'], row['discharge_date']), axis=1
    )
    df_expanded = df_expanded.explode('date_range').reset_index(drop=True)
    df_expanded['date'] = df_expanded['date_range'].dt.strftime('%Y-%m-%d')
    df_expanded['Day_Number'] = (
        df_expanded['date_range'].dt.floor('D') -
        pd.to_datetime(df_expanded['admittime']).dt.floor('D')
    ).dt.days + 1

    # Merge with EHR data
    if 'Day_Number' in df_ehr.columns:
        # Time-varying data (medications): merge on hadm_id and Day_Number
        df_ehr_for_merge = df_ehr[['hadm_id', 'Day_Number', col_name]].copy()
        df_merged = df_expanded.merge(
            df_ehr_for_merge,
            on=['hadm_id', 'Day_Number'],
            how='left'
        )
    else:
        # Non-time-varying data (ICD codes): apply to all days of admission
        df_ehr_for_merge = df_ehr[['hadm_id', col_name]].copy()
        df_merged = df_expanded.merge(
            df_ehr_for_merge,
            on='hadm_id',
            how='left'
        )

    # Create indicator columns and count occurrences
    df_merged[col_name] = df_merged[col_name].fillna('MISSING')

    # Use pandas crosstab for efficient counting (much faster than get_dummies + groupby)
    ct = pd.crosstab(
        [df_merged['subject_id'], df_merged['hadm_id'], df_merged['admittime'],
         df_merged['dischtime'], df_merged['date'], df_merged['target'],
         df_merged['split']],
        df_merged[col_name]
    ).reset_index()

    # Rename columns to use feature values directly
    ct.columns.name = None

    # Create node_name column
    ct['node_name'] = ct['subject_id'].astype(str) + '_' + ct['hadm_id'].astype(str)

    # Reorder columns to match original format
    base_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'date',
                 'target', 'node_name', 'split']
    feature_cols = [col for col in ct.columns if col not in base_cols + ['splits']]

    # Add demographic columns
    demo_data = df_demo[DEMO_COLS + ['hadm_id']].copy()
    for demo in DEMO_COLS:
        if demo in CAT_COLUMNS:
            demo_data[demo] = demo_data[demo].fillna('UNKNOWN')

    # Merge demographics
    ct = ct.merge(demo_data, on='hadm_id', how='left')

    # Final column order (temporarily without 'splits' which we'll add later)
    temp_base_cols = base_cols + DEMO_COLS  # without 'splits'
    df_counts = ct[temp_base_cols + feature_cols]

    # Drop zero-occurrence feature columns and MISSING if present
    if filter_freq is not None:
        freq = df_counts[feature_cols].sum()
        keep_cols = freq[freq >= filter_freq].index.tolist()
    else:
        freq = df_counts[feature_cols].sum()
        keep_cols = freq[freq > 0].index.tolist()

    # Add splits column (duplicate of split for compatibility)
    df_counts['splits'] = df_counts['split']

    # Final column order (now with splits)
    all_base_cols = base_cols + DEMO_COLS + ['splits']
    final_cols = all_base_cols + keep_cols
    df_counts = df_counts[final_cols]

    print(f"Final subgroups: {len(keep_cols)}")

    return df_counts


def lab_one_hot_mimic(df_demo, df_lab, col_name, time_step_by="day", filter_freq=None):
    """
    Vectorized lab one-hot encoding with abnormal flag detection.

    Creates daily indicators for abnormal lab values per admission.
    Uses pandas vectorized operations for 100-1000x speedup over nested loops.

    Args:
        df_demo: demographics dataframe
        df_lab: labs dataframe with hadm_id, Day_Number, label_fluid, flag
        col_name: column name containing lab test identifiers (e.g., 'label_fluid')
        time_step_by: only 'day' is supported
        filter_freq: minimum frequency threshold for abnormal labs (optional)

    Returns:
        DataFrame with one row per admission-day, with abnormal/normal flags per lab
    """
    import warnings
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    # Get unique lab tests
    lab_cols = df_lab[col_name].unique()
    lab_cols = [col for col in lab_cols if isinstance(col, str)]
    print(f"Processing {len(lab_cols)} unique lab tests...")

    # Prepare demographics: add admit/discharge dates and create date ranges
    df_demo = df_demo.copy()
    df_demo['admit_date'] = pd.to_datetime(df_demo['admittime']).dt.date
    df_demo['discharge_date'] = pd.to_datetime(df_demo['dischtime']).dt.date

    # Create expanded dataframe: one row per admission-day
    df_expanded = df_demo.copy()
    df_expanded['date_range'] = df_expanded.apply(
        lambda row: pd.date_range(row['admit_date'], row['discharge_date']), axis=1
    )
    df_expanded = df_expanded.explode('date_range').reset_index(drop=True)
    df_expanded['date'] = df_expanded['date_range'].dt.strftime('%Y-%m-%d')
    df_expanded['Day_Number'] = (
        df_expanded['date_range'].dt.floor('D') -
        pd.to_datetime(df_expanded['admittime']).dt.floor('D')
    ).dt.days + 1

    # Merge with lab data (which already has Day_Number)
    df_labs_for_merge = df_lab[['hadm_id', 'Day_Number', col_name, 'flag']].copy()
    df_merged = df_expanded.merge(
        df_labs_for_merge,
        on=['hadm_id', 'Day_Number'],
        how='left'
    )

    # Create pivot table: one column per lab test with flag values
    df_merged[col_name] = df_merged[col_name].fillna('MISSING')
    df_merged['flag'] = df_merged['flag'].fillna('nan')

    # Use pivot_table to get one row per admission-day and one column per lab
    # aggfunc='first' takes the first occurrence (if multiple same labs in one day)
    pivoted = df_merged.pivot_table(
        index=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'date', 'target', 'split'],
        columns=col_name,
        values='flag',
        aggfunc='first'
    ).reset_index()

    # Flatten column names
    pivoted.columns.name = None

    # Create node_name column
    pivoted['node_name'] = pivoted['subject_id'].astype(str) + '_' + pivoted['hadm_id'].astype(str)

    # Add demographic columns
    demo_data = df_demo[DEMO_COLS + ['hadm_id']].copy()
    for demo in DEMO_COLS:
        if demo in CAT_COLUMNS:
            demo_data[demo] = demo_data[demo].fillna('UNKNOWN')

    # Merge demographics
    pivoted = pivoted.merge(demo_data, on='hadm_id', how='left')

    # Reorder columns (temporarily without 'splits')
    base_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'date',
                 'target', 'node_name', 'split']
    temp_base_cols = base_cols + DEMO_COLS  # without 'splits'
    lab_value_cols = [col for col in pivoted.columns if col not in temp_base_cols]

    df_result = pivoted[temp_base_cols + lab_value_cols]

    # Fill missing lab values with 'nan' (indicating no test that day)
    for col in lab_value_cols:
        df_result[col] = df_result[col].fillna('nan')

    # Drop zero-occurrence abnormal labs (but keep 'nan' values)
    if filter_freq is not None:
        freq = (df_result[lab_value_cols] == 'abnormal').sum()
        keep_cols = freq[freq >= filter_freq].index.tolist()
    else:
        freq = (df_result[lab_value_cols] == 'abnormal').sum()
        keep_cols = freq[freq > 0].index.tolist()

    # Add splits column
    df_result['splits'] = df_result['split']

    # Final column order (now with splits)
    all_base_cols = base_cols + DEMO_COLS + ['splits']
    final_cols = all_base_cols + keep_cols
    df_final = df_result[final_cols]

    initial_cols = len(base_cols) + len(DEMO_COLS) + 2  # +2 for splits and node_name

    print(f"Final labs: {len(keep_cols)}")

    return df_final


def preproc_ehr_cat_embedding(X):

    train_indices = X[X["split"] == "train"].index
    target = "target"

    types = X.dtypes

    # encode categorical variables
    categorical_columns = []
    categorical_dims = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            continue
        if col in CAT_COLUMNS:
            l_enc = LabelEncoder()
            X[col] = X[col].fillna("VV_likely")
            X[col] = X[col].replace(
                {0.23990602999011235: "UNKNOWN"}
            )  # TODO: confirm if removing this works
            print(col, X[col].unique())
            X[col] = l_enc.fit_transform(X[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            print(col)
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)

    feature_cols = [
        col for col in X.columns if (col != target) and (col not in COLS_IRRELEVANT)
    ]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f]
        for i, f in enumerate(feature_cols)
        if f in categorical_columns
    ]

    return {
        "X": X,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def preproc_ehr(X):
    """
    Args:
        X: pandas dataframe
    Returns:
        X_enc: pandas dataframe, with one-hot encoded columns for categorical variables
    """
    train_indices = X[X["split"] == "train"].index

    # encode categorical variables
    X_enc = []
    num_cols = 0
    categorical_columns = []
    categorical_dims = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            X_enc.append(X[col])
            num_cols += 1
        elif col in CAT_COLUMNS:
            print(col, X[col].unique())
            curr_enc = pd.get_dummies(
                X[col], prefix=col
            )  # this will transform into one-hot encoder
            X_enc.append(curr_enc)
            num_cols += curr_enc.shape[-1]
            categorical_columns.append(col)
            categorical_dims[col] = curr_enc.shape[-1]
        else:
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)
            curr_enc = X[col]
            X_enc.append(curr_enc)
            num_cols += 1

    X_enc = pd.concat(X_enc, axis=1)
    assert num_cols == X_enc.shape[-1]

    feature_cols = [
        col
        for col in X_enc.columns
        if (col != "target") and (col not in COLS_IRRELEVANT)
    ]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f]
        for _, f in enumerate(feature_cols)
        if f in categorical_columns
    ]

    return {
        "X": X_enc,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def ehr2sequence(preproc_dict, df_demo, by="day"):
    """
    Arrange EHR into sequences for temporal models
    """
    X = preproc_dict["X"]
    df = copy.deepcopy(X)
    feature_cols = preproc_dict["feature_cols"]

    print("Rearranging to sequences by {}...".format(by))
    X = X[feature_cols].values

    X_dict = {}
    for i in range(X.shape[0]):
        key = (
            str(df.iloc[i]["subject_id"])
            + "_"
            + str(pd.to_datetime(df.iloc[i]["date"]).date())
        )
        X_dict[key] = X[i]

    # Build node_name -> list of day indices using admittime/dischtime rather than imaging metadata
    node_included_files = {}
    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        node_name = f"{row['subject_id']}_{row['hadm_id']}"
        if node_name in node_included_files:
            continue
        adm = pd.to_datetime(row["admittime"]).date()
        dis = pd.to_datetime(row["dischtime"]).date()
        days = pd.date_range(start=adm, end=dis)
        node_included_files[node_name] = [d.date() for d in days]

    # arrange X by day
    feat_dict = {}
    for node_name, days in tqdm(node_included_files.items()):
        subj, hadm = node_name.split("_")
        curr_features = []
        for dt in days:
            key = str(subj) + "_" + str(dt)
            feat = X_dict[key]
            curr_features.append(feat)

        curr_features = np.stack(curr_features)  # (num_days, feature_dim)
        feat_dict[node_name] = curr_features

    if "cat_idxs" in preproc_dict:
        cat_idxs = preproc_dict["cat_idxs"]
        cat_dims = preproc_dict["cat_dims"]
        return {
            "feat_dict": feat_dict,
            "feature_cols": feature_cols,
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
        }
    else:
        return {"feat_dict": feat_dict, "feature_cols": feature_cols}


def load_cohort(path: str) -> pd.DataFrame:
    """Load cohort CSV with expected date parsing."""
    return pd.read_csv(path, parse_dates=["admittime", "dischtime"])


def main(args):
    df_demo = load_cohort(args.demo_file)
    # Handle target column naming
    if "readmitted_within_30days" not in df_demo.columns and "readmitted_within_window" in df_demo.columns:
        df_demo["readmitted_within_30days"] = df_demo["readmitted_within_window"]
    # Also create 'target' alias for internal use
    df_demo = df_demo.copy()
    if "target" not in df_demo.columns:
        if "readmitted_within_30days" in df_demo.columns:
            df_demo["target"] = df_demo["readmitted_within_30days"]
        elif "readmitted_within_window" in df_demo.columns:
            df_demo["target"] = df_demo["readmitted_within_window"]
    if "split" not in df_demo.columns:
        df_demo["split"] = "train"
    if "splits" not in df_demo.columns:
        df_demo["splits"] = df_demo["split"]
    os.makedirs(args.save_dir, exist_ok=True)

    df_icd = preprocess_icd(args.icd_file, df_demo)
    df_lab = preprocess_lab(args.lab_file, df_demo) if not args.skip_labs else None
    df_med = preprocess_med(args.med_file, df_demo) if not args.skip_meds else None

    # icd
    df_icd_count = ehr_bag_of_words_mimic(
        df_demo, df_icd, col_name="SUBGROUP", time_step_by="day", filter_freq=None
    )

    # lab
    if df_lab is not None:
        df_lab_onehot = lab_one_hot_mimic(
            df_demo, df_lab, col_name="label_fluid", time_step_by="day", filter_freq=None
        )
    else:
        df_lab_onehot = None

    # medication
    if df_med is not None:
        df_med_count = ehr_bag_of_words_mimic(
            df_demo,
            df_med,
            col_name="MED_THERAPEUTIC_CLASS_DESCRIPTION",
            time_step_by="day",
            filter_freq=None,
        )
    else:
        df_med_count = None

    # combine
    parts = [df_icd_count]
    if df_lab_onehot is not None:
        parts.append(df_lab_onehot)
    if df_med_count is not None:
        parts.append(df_med_count)

    df_combined = pd.concat(parts, axis=1)

    # drop duplicated columns
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_combined.to_csv(os.path.join(args.save_dir, "ehr_combined.csv"), index=False)

    for format in ["cat_embedding", "one_hot"]:
        if format == "cat_embedding":
            preproc_dict = preproc_ehr_cat_embedding(df_combined)
        else:
            preproc_dict = preproc_ehr(df_combined)

        feature_cols = preproc_dict["feature_cols"]
        demo_cols = [
            col for col in feature_cols if any([s for s in DEMO_COLS if s in col])
        ]
        icd_cols = [
            col for col in feature_cols if col in list(set(df_icd["SUBGROUP"].tolist()))
        ]
        lab_cols = (
            [col for col in feature_cols if any([s for s in LAB_COLS if s in col])]
            if df_lab is not None
            else []
        )
        med_cols = (
            [
                col
                for col in feature_cols
                if col in list(set(df_med["MED_THERAPEUTIC_CLASS_DESCRIPTION"].tolist()))
            ]
            if df_med is not None
            else []
        )

        preproc_dict["demo_cols"] = demo_cols
        preproc_dict["icd_cols"] = icd_cols
        preproc_dict["lab_cols"] = lab_cols
        preproc_dict["med_cols"] = med_cols

        # save
        with open(
            os.path.join(args.save_dir, "ehr_preprocessed_all_{}.pkl".format(format)),
            "wb",
        ) as pf:
            pickle.dump(preproc_dict, pf)
        print(
            "Saved to {}".format(
                os.path.join(
                    args.save_dir, "ehr_preprocessed_all_{}.pkl".format(format)
                )
            )
        )

        # also save it into sequences for temporal models
        seq_dict = ehr2sequence(preproc_dict, df_demo, by="day")

        seq_dict["demo_cols"] = demo_cols
        seq_dict["icd_cols"] = icd_cols
        seq_dict["lab_cols"] = lab_cols
        seq_dict["med_cols"] = med_cols
        with open(
            os.path.join(
                args.save_dir, "ehr_preprocessed_seq_by_day_{}.pkl".format(format)
            ),
            "wb",
        ) as pf:
            pickle.dump(seq_dict, pf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing EHR.")

    parser.add_argument(
        "--demo_file",
        type=str,
        default=None,
        help="Dir to filtered cohort demographics file.",
    )
    parser.add_argument(
        "--icd_file",
        type=str,
        default=None,
        help="Dir to filtered cohort ICD-10 file.",
    )
    parser.add_argument(
        "--lab_file",
        type=str,
        default=None,
        help="Dir to filtered cohort lab file.",
    )
    parser.add_argument(
        "--med_file",
        type=str,
        default=None,
        help="Dir to filtered cohort medication file.",
    )
    parser.add_argument(
        "--skip-labs",
        action="store_true",
        help="Skip processing labevents for faster smoke tests.",
    )
    parser.add_argument(
        "--skip-meds",
        action="store_true",
        help="Skip processing prescriptions for faster smoke tests.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Dir to save preprocessed files."
    )

    args = parser.parse_args()
    main(args)
