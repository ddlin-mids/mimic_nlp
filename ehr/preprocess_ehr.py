import argparse
import copy
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Any, List

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


def preprocess_med(path: str, df_demo: pd.DataFrame, ndc_map_path: str) -> pd.DataFrame:
    """
    Load meds for the cohort admissions and derive Day_Number + therapeutic class.
    """
    hadm_set = set(df_demo["hadm_id"])
    adm_map = df_demo.set_index("hadm_id")["admittime"].to_dict()
    dis_map = df_demo.set_index("hadm_id")["dischtime"].to_dict()
    los_map = df_demo.set_index("hadm_id")["length_of_stay_days"].to_dict()

    # load ndc -> therapeutic class map if available
    ndc_map = {}
    if os.path.exists(ndc_map_path):
        ndc_df = pd.read_csv(ndc_map_path)
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
    df_demo: pd.DataFrame, df_ehr: pd.DataFrame, col_name: str, time_step_by: str = "day", filter_freq: Optional[int] = None
) -> pd.DataFrame:
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


def lab_one_hot_mimic(df_demo: pd.DataFrame, df_lab: pd.DataFrame, col_name: str, time_step_by: str = "day", filter_freq: Optional[int] = None) -> pd.DataFrame:
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


def preproc_ehr_cat_embedding(X: pd.DataFrame) -> Dict[str, Any]:

    train_indices = X[X["split"] == "train"].index
    target = "target"

    # Identify columns to encode
    # We encode explicit CAT_COLUMNS or any object/string column that is not metadata
    cols_to_encode = []
    for col in X.columns:
        if col in COLS_IRRELEVANT or col == target:
            continue
        if col in CAT_COLUMNS or X[col].dtype == object:
            cols_to_encode.append(col)
    
    print(f"Encoding {len(cols_to_encode)} categorical columns...")
    
    categorical_dims = {}
    # Process categorical columns (LabelEncoder)
    # This loop is necessary as LabelEncoder is per-column
    import gc
    for i, col in enumerate(tqdm(cols_to_encode, desc="Label Encoding")):
        l_enc = LabelEncoder()
        X[col] = X[col].fillna("VV_likely")
        X[col] = X[col].replace({0.23990602999011235: "UNKNOWN"})
        # Ensure string type for robust encoding
        X[col] = l_enc.fit_transform(X[col].astype(str).values)
        categorical_dims[col] = len(l_enc.classes_)
        
        # Explicit GC every 50 iterations to prevent fragmentation
        if i % 50 == 0:
            gc.collect()

    # Identify numerical columns (everything else)
    feature_cols = [c for c in X.columns if c not in COLS_IRRELEVANT and c != target]
    num_cols = [c for c in feature_cols if c not in cols_to_encode]
    
    print(f"Filling missing values for {len(num_cols)} numerical columns with 0...")
    
    # Vectorized FillNA for numerical columns (counts)
    # For ICD/Meds counts, missing means 0.
    if num_cols:
        # Process in chunks to avoid OOM
        chunk_size = 50
        for i in range(0, len(num_cols), chunk_size):
            c_cols = num_cols[i : i + chunk_size]
            X[c_cols] = X[c_cols].fillna(0)

    feature_cols = [col for col in X.columns if (col != target) and (col not in COLS_IRRELEVANT)]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in cols_to_encode]
    cat_dims = [categorical_dims[f] for f in feature_cols if f in cols_to_encode]

    return {
        "X": X,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def preproc_ehr(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Args:
        X: pandas dataframe
    Returns:
        X_enc: pandas dataframe, with one-hot encoded columns for categorical variables
    """
    train_indices = X[X["split"] == "train"].index
    target = "target"

    # Identify types of columns
    cat_cols = [] # Standard demographics
    lab_cols = [] # Lab flags ("abnormal", "nan")
    num_cols = [] # Counts
    
    for col in X.columns:
        if col in COLS_IRRELEVANT or col == target:
            continue
        
        # Check if it's a lab column (heuristically, if it contains "abnormal")
        # A safer check: check if it's in the object list but NOT in CAT_COLUMNS
        if col in CAT_COLUMNS:
            cat_cols.append(col)
        elif X[col].dtype == object:
            # Ideally we track lab cols explicitly, but here we infer
            # If it's an object column not in CAT_COLUMNS (like gender/race), it's likely a lab flag column
            lab_cols.append(col)
        else:
            num_cols.append(col)

    print(f"Encoding: {len(cat_cols)} cat, {len(lab_cols)} labs, {len(num_cols)} numeric...")
    
    X_enc_parts = [X[COLS_IRRELEVANT + [target] if target in X.columns else COLS_IRRELEVANT]]
    
    # 1. Standard Categoricals (Gender, Race) -> One-Hot (drop_first=True)
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna("UNKNOWN")
        dummies = pd.get_dummies(X[cat_cols], prefix=cat_cols, drop_first=True)
        X_enc_parts.append(dummies)
    
    # 2. Lab Columns -> Binary (Abnormal=1, else 0)
    if lab_cols:
        print("Binarizing lab flags...")
        # This is a massive vectorized op: (X[lab_cols] == 'abnormal').astype(int)
        # Note: 'nan' (string) and actual NaN/None will both be False.
        lab_binary = (X[lab_cols] == 'abnormal').astype(int)
        # Rename columns to indicate abnormality
        lab_binary.columns = [f"{c}_abnormal" for c in lab_binary.columns]
        X_enc_parts.append(lab_binary)
        
    # 3. Numerical Columns -> FillNA(0)
    if num_cols:
        print("Filling numericals with 0...")
        X_filled_num = X[num_cols].fillna(0)
        X_enc_parts.append(X_filled_num)

    X_enc = pd.concat(X_enc_parts, axis=1)
    
    feature_cols = [col for col in X_enc.columns if (col != "target") and (col not in COLS_IRRELEVANT)]
    
    return {
        "X": X_enc,
        "feature_cols": feature_cols,
        "cat_idxs": [],
        "cat_dims": [],
    }


def ehr2sequence(preproc_dict: Dict[str, Any], df_demo: pd.DataFrame, by: str = "day") -> Dict[str, Any]:
    """
    Arrange EHR into sequences for temporal models using memory-efficient grouping.
    """
    import gc
    
    X = preproc_dict["X"]
    feature_cols = preproc_dict["feature_cols"]
    
    print(f"Rearranging to sequences by {by} (memory optimized)...")
    
    # 1. Ensure we only work with necessary data
    # Create a lightweight view with just the indices and features
    cols_to_keep = ['subject_id', 'hadm_id', 'date'] + feature_cols
    df_slim = X[cols_to_keep].copy()
    
    # Convert date to string format for consistency if needed, but keeping as object/datetime is fine for sorting
    df_slim['date'] = pd.to_datetime(df_slim['date'])
    
    # Sort by hadm_id and date to ensure correct temporal order
    df_slim = df_slim.sort_values(['subject_id', 'hadm_id', 'date'])
    
    # Create the mapping key in df_slim
    df_slim['node_name'] = df_slim['subject_id'].astype(str) + "_" + df_slim['hadm_id'].astype(str)
    
    # Extract the feature matrix (this is the heavy part)
    # We do this AFTER sorting so it aligns with the dataframe
    print("Extracting feature matrix...")
    feature_matrix = df_slim[feature_cols].values
    if feature_matrix.dtype == np.float64:
         feature_matrix = feature_matrix.astype(np.float32)
    
    # Efficiently group by node_name
    print("Grouping by admission...")
    grouped = df_slim.groupby('node_name', sort=False)
    
    feat_dict = {}
    
    # Iterate through groups - this is safe if we don't materialize everything at once
    # tqdm wrapper for progress
    for node_name, group_idxs in tqdm(grouped.indices.items(), desc="Building sequences"):
        # group_idxs is an array of integer indices into df_slim/feature_matrix
        # Extract the slice from the matrix
        seq = feature_matrix[group_idxs]
        feat_dict[node_name] = seq

    # Clean up massive intermediates
    del df_slim
    del feature_matrix
    del grouped
    gc.collect()

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


def process_chunk(chunk_id: int, df_chunk: pd.DataFrame, df_icd: Optional[pd.DataFrame], df_lab: Optional[pd.DataFrame], df_med: Optional[pd.DataFrame], global_icd_cols: Set[str], global_lab_cols: Set[str], global_med_cols: Set[str], args: argparse.Namespace) -> pd.DataFrame:
    """Process a single chunk of admissions."""
    # Filter EHR data for this chunk
    chunk_hadms = set(df_chunk["hadm_id"])
    
    # ICD
    df_icd_count = None
    if df_icd is not None:
        df_icd_chunk = df_icd[df_icd["hadm_id"].isin(chunk_hadms)].copy()
        df_icd_count = ehr_bag_of_words_mimic(
            df_chunk, df_icd_chunk, col_name="SUBGROUP", time_step_by="day", filter_freq=None
        )
        # Ensure all global cols exist
        if global_icd_cols:
            # Use reindex to add missing columns with 0, keeping existing ones
            # Note: This preserves index and existing data
            existing_cols = [c for c in df_icd_count.columns if c in global_icd_cols]
            missing_cols = [c for c in global_icd_cols if c not in df_icd_count.columns]
            if missing_cols:
                df_icd_count[missing_cols] = 0
    
    # Lab
    df_lab_onehot = None
    if df_lab is not None:
        df_lab_chunk = df_lab[df_lab["hadm_id"].isin(chunk_hadms)].copy()
        df_lab_onehot = lab_one_hot_mimic(
            df_chunk, df_lab_chunk, col_name="label_fluid", time_step_by="day", filter_freq=None
        )
        # Ensure all global cols exist
        if global_lab_cols:
            missing_cols = [c for c in global_lab_cols if c not in df_lab_onehot.columns]
            if missing_cols:
                df_lab_onehot[missing_cols] = "nan"
        
    # Med
    df_med_count = None
    if df_med is not None:
        df_med_chunk = df_med[df_med["hadm_id"].isin(chunk_hadms)].copy()
        df_med_count = ehr_bag_of_words_mimic(
            df_chunk,
            df_med_chunk,
            col_name="MED_THERAPEUTIC_CLASS_DESCRIPTION",
            time_step_by="day",
            filter_freq=None,
        )
        # Ensure all global cols exist
        if global_med_cols:
            missing_cols = [c for c in global_med_cols if c not in df_med_count.columns]
            if missing_cols:
                df_med_count[missing_cols] = 0
        
    # Combine
    parts = []
    if df_icd_count is not None: parts.append(df_icd_count)
    if df_lab_onehot is not None: parts.append(df_lab_onehot)
    if df_med_count is not None: parts.append(df_med_count)
        
    if not parts:
        return df_chunk # Should not happen normally
        
    df_chunk_combined = pd.concat(parts, axis=1)
    df_chunk_combined = df_chunk_combined.loc[:, ~df_chunk_combined.columns.duplicated()]
    
    return df_chunk_combined

def augment_cohort_with_history(cohort: pd.DataFrame, admissions_path: str) -> pd.DataFrame:
    """
    Calculates prior admission history features:
    1. num_prior_admissions: Total previous admissions
    2. num_prior_30d_readmissions: Number of times readmitted within 30 days previously
    3. days_since_last_discharge: Gap since previous discharge
    """
    print(f"Loading admissions from {admissions_path}...")
    # We only need columns to calculate timing
    cols = ["subject_id", "hadm_id", "admittime", "dischtime"]
    all_adm = pd.read_csv(admissions_path, usecols=cols, parse_dates=["admittime", "dischtime"])
    
    # Filter to subjects in our cohort to save memory
    cohort_subjects = set(cohort["subject_id"].unique())
    all_adm = all_adm[all_adm["subject_id"].isin(cohort_subjects)].copy()
    
    # Sort by subject and time
    all_adm = all_adm.sort_values(["subject_id", "admittime"])
    
    # Calculate previous discharge time
    all_adm["prev_dischtime"] = all_adm.groupby("subject_id")["dischtime"].shift(1)
    all_adm["days_since_last"] = (all_adm["admittime"] - all_adm["prev_dischtime"]).dt.total_seconds() / (24 * 3600)
    
    # Identify 30-day readmissions (gap < 30 days and > 0)
    all_adm["is_30d_readmit"] = (all_adm["days_since_last"] <= 30) & (all_adm["days_since_last"] > 0)
    
    # Cumulative sums
    # 1. Total prior admissions (cumcount)
    all_adm["num_prior_admissions"] = all_adm.groupby("subject_id").cumcount()
    
    # 2. Total prior 30-day readmissions
    all_adm["num_prior_30d_readmissions"] = all_adm.groupby("subject_id")["is_30d_readmit"].cumsum() - all_adm["is_30d_readmit"].astype(int)
    
    # Select relevant columns to merge
    features = all_adm[["hadm_id", "num_prior_admissions", "num_prior_30d_readmissions", "days_since_last"]]
    
    # Merge back into cohort
    print("Merging history features...")
    # Use left join to keep only cohort rows
    cohort = cohort.merge(features, on="hadm_id", how="left")
    
    # Fill NaNs for days_since_last (first admission has no gap)
    # We can use -1 to indicate "never"
    cohort["days_since_last_discharge"] = cohort["days_since_last"].fillna(-1)
    cohort.drop(columns=["days_since_last"], inplace=True)
    
    return cohort

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggressively optimize memory usage by:
    1. Converting object columns that should be numeric to float32
    2. Downcasting float64/int64 to float32/int32
    """
    import gc
    print("Optimizing dataframe memory...")
    start_mem = df.memory_usage().sum() / 1024**3
    print(f"Initial memory usage: {start_mem:.2f} GB")

    # Identify metadata/categorical columns to exclude from numeric conversion
    # We use global variables or heuristic
    keep_obj_cols = set(COLS_IRRELEVANT + CAT_COLUMNS + ["target", "split", "splits"])
    
    # 1. Fix mixed-type object columns (from DtypeWarning)
    # Filter for object columns that are NOT in our known categorical list
    obj_cols = [c for c in df.columns if df[c].dtype == object and c not in keep_obj_cols]
    
    if obj_cols:
        print(f"Attempting to convert {len(obj_cols)} object columns to numeric...")
        # Process in chunks to check progress
        chunk_size = 100
        for i in range(0, len(obj_cols), chunk_size):
            chunk = obj_cols[i:i+chunk_size]
            # coerce errors to NaN (which will be 0 later), then downcast
            df[chunk] = df[chunk].apply(pd.to_numeric, errors='coerce').astype(np.float32)
    
    # 2. Downcast numeric columns
    # Float64 -> Float32
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        print(f"Downcasting {len(float_cols)} float64 columns to float32...")
        df[float_cols] = df[float_cols].astype(np.float32)

    # Int64 -> Int32
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(int_cols) > 0:
        print(f"Downcasting {len(int_cols)} int64 columns to int32...")
        df[int_cols] = df[int_cols].astype(np.int32)
        
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024**3
    print(f"Final memory usage: {end_mem:.2f} GB (Saved {start_mem - end_mem:.2f} GB)")
    return df

def main(args: argparse.Namespace):
    import gc
    
    df_demo = load_cohort(args.demo_file)
    
    # Augment with history features
    df_demo = augment_cohort_with_history(df_demo, args.admissions_file)
    
    # Add new features to global DEMO_COLS so they are preserved
    global DEMO_COLS
    
    # 1. History features (calculated above)
    new_features = ["num_prior_admissions", "num_prior_30d_readmissions", "days_since_last_discharge"]
    
    # 2. Cohort specific features (conditions, admission info)
    cohort_features = [
        "admission_type", "discharge_location", 
        "is_cardiorenal_long", "has_acute_kidney_injury", "has_heart_failure", 
        "has_hyponatremia", "has_posthemorrhagic_anemia", "has_sepsis", 
        "has_any_cardiorenal_sepsis", "has_aki_and_hf"
    ]
    
    # Only add if not already present to avoid duplicates on re-runs
    for f in new_features + cohort_features:
        if f in df_demo.columns and f not in DEMO_COLS:
            DEMO_COLS.append(f)
            # If string/categorical, ensure it's in CAT_COLUMNS for proper encoding
            if df_demo[f].dtype == object and f not in CAT_COLUMNS:
                CAT_COLUMNS.append(f)
            
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
    temp_chunk_dir = os.path.join(args.save_dir, "temp_chunks")
    os.makedirs(temp_chunk_dir, exist_ok=True)

    print("Loading raw EHR files...")
    df_icd = preprocess_icd(args.icd_file, df_demo)
    df_lab = preprocess_lab(args.lab_file, df_demo) if not args.skip_labs else None
    df_med = preprocess_med(args.med_file, df_demo, args.ndc_map_file) if not args.skip_meds else None
    print("Raw EHR files loaded.")
    
    # Pre-compute global features to ensure consistency across chunks
    global_icd_cols = set(df_icd["SUBGROUP"].unique()) if df_icd is not None else set()
    global_icd_cols = {c for c in global_icd_cols if isinstance(c, str) and c not in SUBGOUPRS_EXCLUDED}
    
    global_lab_cols = set(df_lab["label_fluid"].unique()) if df_lab is not None else set()
    global_lab_cols = {c for c in global_lab_cols if isinstance(c, str)}
    
    global_med_cols = set(df_med["MED_THERAPEUTIC_CLASS_DESCRIPTION"].unique()) if df_med is not None else set()
    global_med_cols = {c for c in global_med_cols if isinstance(c, str) and c not in SUBGOUPRS_EXCLUDED}
    
    print(f"Global features identified: {len(global_icd_cols)} ICD groups, {len(global_lab_cols)} Labs, {len(global_med_cols)} Med classes.")

    combined_csv_path = os.path.join(args.save_dir, "ehr_combined.csv")
    if os.path.exists(combined_csv_path):
        print(f"Found existing combined EHR data at {combined_csv_path}. Loading...")
        # Use low_memory=False to ensure cleaner initial read despite memory cost, 
        # because we will immediately optimize it down.
        df_combined = pd.read_csv(combined_csv_path, low_memory=False)
        print(f"Loaded combined shape: {df_combined.shape}")
        
        # Optimize memory immediately
        df_combined = optimize_dataframe_memory(df_combined)
    else:
        # Processing in chunks
        chunk_size = args.chunk_size
        num_chunks = (len(df_demo) + chunk_size - 1) // chunk_size
        print(f"Processing {len(df_demo)} admissions in {num_chunks} chunks (size={chunk_size})...")

        chunk_files = []
        
        for i in tqdm(range(num_chunks), desc="Chunks"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df_demo))
            
            chunk_file = os.path.join(temp_chunk_dir, f"chunk_{i}_{start_idx}_{end_idx}.pkl")
            chunk_files.append(chunk_file)
            
            if os.path.exists(chunk_file):
                print(f"Chunk {i} already exists, skipping...")
                continue
                
            print(f"Processing chunk {i} ({start_idx} to {end_idx})...", flush=True)
            df_chunk = df_demo.iloc[start_idx:end_idx].copy()
            
            try:
                df_chunk_res = process_chunk(
                    i, df_chunk, df_icd, df_lab, df_med, 
                    global_icd_cols, global_lab_cols, global_med_cols, 
                    args
                )
                
                with open(chunk_file, "wb") as f:
                    pickle.dump(df_chunk_res, f)
                
                del df_chunk_res
                del df_chunk
                gc.collect()
                print(f"[Progress] Finished chunk {i+1}/{num_chunks}", flush=True)
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}", flush=True)
                raise e

        print("Aggregating chunks...")
        all_chunks = []
        for cf in tqdm(chunk_files, desc="Loading Chunks"):
            with open(cf, "rb") as f:
                all_chunks.append(pickle.load(f))
                
        df_combined = pd.concat(all_chunks, axis=0, ignore_index=True)
        del all_chunks
        gc.collect()
        
        print(f"Final combined shape: {df_combined.shape}")

        # drop duplicated columns (again, just in case)
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
        
        # Save full combined raw CSV
        df_combined.to_csv(combined_csv_path, index=False)

    formats = []
    if not args.skip_cat_embedding:
        formats.append("cat_embedding")
    if not args.skip_one_hot:
        formats.append("one_hot")

    for i, format in enumerate(formats):
        out_file = os.path.join(args.save_dir, "ehr_preprocessed_all_{}.pkl".format(format))
        seq_out_file = os.path.join(args.save_dir, "ehr_preprocessed_seq_by_day_{}.pkl".format(format))

        if os.path.exists(out_file) and os.path.exists(seq_out_file):
            print(f"[{format}] Artifacts already exist. Skipping generation.")
            continue

        print(f"Generating {format} representation...")
        # Avoid copy for the last iteration to save memory
        is_last = (i == len(formats) - 1)
        df_input = df_combined if is_last else df_combined.copy()

        if format == "cat_embedding":
            preproc_dict = preproc_ehr_cat_embedding(df_input)
        else:
            preproc_dict = preproc_ehr(df_input)

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
        with open(out_file, "wb") as pf:
            pickle.dump(preproc_dict, pf)
        print(f"Saved to {out_file}")

        # also save it into sequences for temporal models
        seq_dict = ehr2sequence(preproc_dict, df_demo, by="day")

        seq_dict["demo_cols"] = demo_cols
        seq_dict["icd_cols"] = icd_cols
        seq_dict["lab_cols"] = lab_cols
        seq_dict["med_cols"] = med_cols
        
        with open(seq_out_file, "wb") as pf:
            pickle.dump(seq_dict, pf)
        print(f"Saved sequences to {seq_out_file}")


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
    parser.add_argument(
        "--admissions_file",
        type=str,
        default="physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
        help="Path to raw admissions file for history calculation.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=2000, help="Number of admissions to process per chunk."
    )
    parser.add_argument(
        "--ndc_map_file",
        type=str,
        default=str(REPO_ROOT / "refs/readmit-stgnn/data/ndc2therapeutic.csv"),
        help="Path to NDC to therapeutic class mapping file.",
    )

    parser.add_argument(
        "--skip-cat-embedding",
        action="store_true",
        help="Skip generating categorical embedding (LabelEncoded) artifacts.",
    )
    parser.add_argument(
        "--skip-one-hot",
        action="store_true",
        help="Skip generating one-hot encoded artifacts.",
    )

    args = parser.parse_args()
    main(args)

    