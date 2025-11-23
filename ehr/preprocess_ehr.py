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
    Load labs for the cohort admissions and derive Day_Number.
    label_fluid is set to the itemid string; flag is passed through.
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
            flag=chunk["flag"].fillna("nan"),
            Day_Number=chunk["day_num"].astype(float),
        )
        frames.append(chunk[["hadm_id", "label_fluid", "flag", "Day_Number"]])
    if not frames:
        return pd.DataFrame(columns=["hadm_id", "label_fluid", "flag", "Day_Number"])
    return pd.concat(frames, ignore_index=True)


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
    Get EHR sequence using naive bag-of-words method
    Args:
        df_demo: demographics dataframe
        df_ehr: CPT/ICD dataframe
        ehr_type: 'cpt' or 'icd'
        time_step_by: 'day', what is the time step size?
    Returns:
        ehr_seq_padded: shape (num_admissions, max_seq_len, num_ehr_subgroups),
            short sequences are padded with -1
    """

    all_values = list(set(df_ehr[col_name]))
    all_values = [
        val
        for val in all_values
        if isinstance(val, str) and (val not in SUBGOUPRS_EXCLUDED)
    ]

    df_ehr_count = {
        "subject_id": [],
        "hadm_id": [],
        "admittime": [],
        "dischtime": [],
        "date": [],
        "target": [],
        "node_name": [],
        "split": [],
    }
    initial_cols = len(df_ehr_count) + len(DEMO_COLS)
    # add demographic columns
    for demo in DEMO_COLS:
        df_ehr_count[demo] = []

    # add subgroup columns
    for subgrp in all_values:
        df_ehr_count[subgrp] = []

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["subject_id"]
        admit_id = row["hadm_id"]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        label = row["readmitted_within_30days"]

        if (str(pat) + "_" + str(admit_id)) in df_ehr_count["node_name"]:
            continue

        if time_step_by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(admit_dt).date(),
                end=pd.to_datetime(discharge_dt).date(),
            )  # both inclusive
        else:
            raise NotImplementedError
        assert len(dt_range) > 1

        curr_ehr_df = df_ehr[df_ehr["hadm_id"] == admit_id]

        for dt in dt_range:
            day_num = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1

            if "charttime" in df_ehr.columns:
                curr_day_ehrs = curr_ehr_df[
                    curr_ehr_df["Day_Number"] == float(day_num)
                ][col_name]
            else:
                # not time-varying, i.e., diagnoses ICD code
                curr_day_ehrs = curr_ehr_df[col_name]

            df_ehr_count["subject_id"].append(pat)
            df_ehr_count["hadm_id"].append(admit_id)
            df_ehr_count["admittime"].append(admit_dt)
            df_ehr_count["dischtime"].append(discharge_dt)
            df_ehr_count["date"].append(str(dt))
            df_ehr_count["target"].append(label)
            df_ehr_count["split"].append(row["split"])
            df_ehr_count["node_name"].append(str(pat) + "_" + str(admit_id))

            for demo in DEMO_COLS:
                if (demo in CAT_COLUMNS) and isinstance(row[demo], float):  # nan
                    df_ehr_count[demo].append("UNKNOWN")
                else:
                    df_ehr_count[demo].append(row[demo])

            if len(curr_day_ehrs) > 0:
                ehr_counts = Counter(curr_day_ehrs)
                for subgrp in all_values:
                    if subgrp in ehr_counts.keys():
                        df_ehr_count[subgrp].append(ehr_counts[subgrp])
                    else:
                        df_ehr_count[subgrp].append(0)
            else:
                for subgrp in all_values:
                    df_ehr_count[subgrp].append(0)

    df_ehr_count = pd.DataFrame.from_dict(df_ehr_count)
    df_ehr_count["splits"] = df_ehr_count["split"]

    # drop zero occurrence subgroups
    if filter_freq is not None:
        freq = df_ehr_count[all_values].sum(axis=0)
        drop_col_idxs = freq.values < filter_freq
        df_ehr_count = df_ehr_count.drop(columns=freq.loc[drop_col_idxs].index)
    else:
        freq = df_ehr_count[all_values].sum(axis=0)
        drop_col_idxs = freq.values == 0
        df_ehr_count = df_ehr_count.drop(columns=freq.loc[drop_col_idxs].index)

    print("Final subgroups:", len(df_ehr_count.columns) - initial_cols)

    return df_ehr_count


def lab_one_hot_mimic(df_demo, df_lab, col_name, time_step_by="day", filter_freq=None):
    """
    Get EHR sequence using naive bag-of-words method
    Args:
        df_demo: demographics dataframe
        df_ehr: CPT/ICD dataframe
        ehr_type: 'cpt' or 'icd'
        time_step_by: 'day', what is the time step size?
    Returns:
        ehr_seq_padded: shape (num_admissions, max_seq_len, num_ehr_subgroups),
            short sequences are padded with -1
    """

    lab_cols = list(set(df_lab[col_name]))
    lab_cols = [col for col in lab_cols if isinstance(col, str)]

    df_lab_onehot = {
        "subject_id": [],
        "hadm_id": [],
        "admittime": [],
        "dischtime": [],
        "date": [],
        "target": [],
        "node_name": [],
        "split": [],
    }
    initial_cols = len(df_lab_onehot) + len(DEMO_COLS)
    # add demographic columns
    for demo in DEMO_COLS:
        df_lab_onehot[demo] = []

    # add subgroup columns
    for col in lab_cols:
        df_lab_onehot[col] = []

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["subject_id"]
        admit_id = row["hadm_id"]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        label = row["readmitted_within_30days"]

        if (str(pat) + "_" + str(admit_id)) in df_lab_onehot["node_name"]:
            continue

        if time_step_by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(admit_dt).date(),
                end=pd.to_datetime(discharge_dt).date(),
            )  # both inclusive
        else:
            raise NotImplementedError
        assert len(dt_range) > 1

        curr_ehr_df = df_lab[df_lab["hadm_id"] == admit_id]

        for dt in dt_range:
            day_num = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1
            curr_day_lab = curr_ehr_df[curr_ehr_df["Day_Number"] == float(day_num)]

            df_lab_onehot["subject_id"].append(pat)
            df_lab_onehot["hadm_id"].append(admit_id)
            df_lab_onehot["admittime"].append(admit_dt)
            df_lab_onehot["dischtime"].append(discharge_dt)
            df_lab_onehot["date"].append(str(dt))
            df_lab_onehot["target"].append(label)
            df_lab_onehot["split"].append(row["split"])
            df_lab_onehot["node_name"].append(str(pat) + "_" + str(admit_id))

            for demo in DEMO_COLS:
                if (demo in CAT_COLUMNS) and isinstance(row[demo], float):  # nan
                    df_lab_onehot[demo].append("UNKNOWN")
                else:
                    df_lab_onehot[demo].append(row[demo])

            for lab in lab_cols:
                if len(curr_day_lab) == 0:
                    df_lab_onehot[lab].append("nan")
                else:
                    if (
                        curr_day_lab.loc[curr_day_lab[col_name] == lab, "flag"]
                        == "abnormal"
                    ).any():
                        df_lab_onehot[lab].append("abnormal")
                    else:
                        df_lab_onehot[lab].append("nan")

    df_lab_onehot = pd.DataFrame.from_dict(df_lab_onehot)
    df_lab_onehot["splits"] = df_lab_onehot["split"]

    # drop zero abnormal subgroups
    if filter_freq is not None:
        freq = (df_lab_onehot[lab_cols] == "abnormal").sum(
            axis=0
        )  # number of abnormals
        drop_col_idxs = freq.values < filter_freq
        df_lab_onehot = df_lab_onehot.drop(columns=freq.loc[drop_col_idxs].index)
    else:
        freq = (df_lab_onehot[lab_cols] == "abnormal").sum(
            axis=0
        )  # number of abnormals
        drop_col_idxs = freq.values == 0
        df_lab_onehot = df_lab_onehot.drop(columns=freq.loc[drop_col_idxs].index)

    print("Final labs:", len(df_lab_onehot.columns) - initial_cols)

    return df_lab_onehot


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
    if "readmitted_within_30days" not in df_demo.columns and "readmitted_within_window" in df_demo.columns:
        df_demo["readmitted_within_30days"] = df_demo["readmitted_within_window"]
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
