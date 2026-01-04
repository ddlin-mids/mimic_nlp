"""
Extract Clinical Risk Scores for Readmission Prediction.

Features extracted:
1. LACE Score Components:
   - L: Length of stay (0-7 points)
   - A: Acuity of admission (0 or 3 points)
   - C: Charlson Comorbidity Index (0-5 points)
   - E: ED visits in past 6 months (0-4 points)
   
2. Charlson Comorbidity Index (full score):
   - 17 weighted comorbidities from ICD codes
   
3. Individual Comorbidity Flags:
   - Useful for ablation studies and interpretability

Usage:
    python src/data/extract_risk_scores.py \
        --cohort_path data/interim/readmit_analysis/long_los_cohort.csv \
        --output_path data/interim/ehr_long_los/risk_scores.csv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Charlson Comorbidity Index ICD-9 and ICD-10 mappings
# Based on Quan et al. 2005 updated coding algorithms
CHARLSON_CODES = {
    "myocardial_infarction": {
        "icd9": ["410", "412"],
        "icd10": ["I21", "I22", "I252"],
        "weight": 1
    },
    "congestive_heart_failure": {
        "icd9": ["398.91", "402.01", "402.11", "402.91", "404.01", "404.03",
                 "404.11", "404.13", "404.91", "404.93", "425.4", "425.5",
                 "425.6", "425.7", "425.8", "425.9", "428"],
        "icd10": ["I09.9", "I11.0", "I13.0", "I13.2", "I25.5", "I42.0",
                  "I42.5", "I42.6", "I42.7", "I42.8", "I42.9", "I43", "I50", "P29.0"],
        "weight": 1
    },
    "peripheral_vascular_disease": {
        "icd9": ["093.0", "437.3", "440", "441", "443.1", "443.2", "443.3",
                 "443.8", "443.9", "447.1", "557.1", "557.9", "V43.4"],
        "icd10": ["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1", "I79.0",
                  "I79.2", "K55.1", "K55.8", "K55.9", "Z95.8", "Z95.9"],
        "weight": 1
    },
    "cerebrovascular_disease": {
        "icd9": ["362.34", "430", "431", "432", "433", "434", "435", "436",
                 "437", "438"],
        "icd10": ["G45", "G46", "H34.0", "I60", "I61", "I62", "I63", "I64",
                  "I65", "I66", "I67", "I68", "I69"],
        "weight": 1
    },
    "dementia": {
        "icd9": ["290", "294.1", "331.2"],
        "icd10": ["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"],
        "weight": 1
    },
    "chronic_pulmonary_disease": {
        "icd9": ["416.8", "416.9", "490", "491", "492", "493", "494", "495",
                 "496", "497", "498", "499", "500", "501", "502", "503",
                 "504", "505", "506.4", "508.1", "508.8"],
        "icd10": ["I27.8", "I27.9", "J40", "J41", "J42", "J43", "J44", "J45",
                  "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65",
                  "J66", "J67", "J68.4", "J70.1", "J70.3"],
        "weight": 1
    },
    "rheumatic_disease": {
        "icd9": ["446.5", "710.0", "710.1", "710.2", "710.3", "710.4",
                 "714.0", "714.1", "714.2", "714.8", "725"],
        "icd10": ["M05", "M06", "M31.5", "M32", "M33", "M34", "M35.1",
                  "M35.3", "M36.0"],
        "weight": 1
    },
    "peptic_ulcer_disease": {
        "icd9": ["531", "532", "533", "534"],
        "icd10": ["K25", "K26", "K27", "K28"],
        "weight": 1
    },
    "mild_liver_disease": {
        "icd9": ["070.22", "070.23", "070.32", "070.33", "070.44", "070.54",
                 "070.6", "070.9", "570", "571", "573.3", "573.4", "573.8",
                 "573.9", "V42.7"],
        "icd10": ["B18", "K70.0", "K70.1", "K70.2", "K70.3", "K70.9", "K71.3",
                  "K71.4", "K71.5", "K71.7", "K73", "K74", "K76.0", "K76.2",
                  "K76.3", "K76.4", "K76.8", "K76.9", "Z94.4"],
        "weight": 1
    },
    "diabetes_without_complications": {
        "icd9": ["250.0", "250.1", "250.2", "250.3", "250.8", "250.9"],
        "icd10": ["E10.0", "E10.1", "E10.6", "E10.8", "E10.9", "E11.0",
                  "E11.1", "E11.6", "E11.8", "E11.9", "E12.0", "E12.1",
                  "E12.6", "E12.8", "E12.9", "E13.0", "E13.1", "E13.6",
                  "E13.8", "E13.9", "E14.0", "E14.1", "E14.6", "E14.8", "E14.9"],
        "weight": 1
    },
    "diabetes_with_complications": {
        "icd9": ["250.4", "250.5", "250.6", "250.7"],
        "icd10": ["E10.2", "E10.3", "E10.4", "E10.5", "E10.7", "E11.2",
                  "E11.3", "E11.4", "E11.5", "E11.7", "E12.2", "E12.3",
                  "E12.4", "E12.5", "E12.7", "E13.2", "E13.3", "E13.4",
                  "E13.5", "E13.7", "E14.2", "E14.3", "E14.4", "E14.5", "E14.7"],
        "weight": 2
    },
    "hemiplegia_paraplegia": {
        "icd9": ["334.1", "342", "343", "344.0", "344.1", "344.2", "344.3",
                 "344.4", "344.5", "344.6", "344.9"],
        "icd10": ["G04.1", "G11.4", "G80.1", "G80.2", "G81", "G82", "G83.0",
                  "G83.1", "G83.2", "G83.3", "G83.4", "G83.9"],
        "weight": 2
    },
    "renal_disease": {
        "icd9": ["403.01", "403.11", "403.91", "404.02", "404.03", "404.12",
                 "404.13", "404.92", "404.93", "582", "583.0", "583.1",
                 "583.2", "583.4", "583.6", "583.7", "585", "586", "588.0",
                 "V42.0", "V45.1", "V56"],
        "icd10": ["I12.0", "I13.1", "N03.2", "N03.3", "N03.4", "N03.5",
                  "N03.6", "N03.7", "N05.2", "N05.3", "N05.4", "N05.5",
                  "N05.6", "N05.7", "N18", "N19", "N25.0", "Z49", "Z94.0", "Z99.2"],
        "weight": 2
    },
    "malignancy": {
        "icd9": ["140", "141", "142", "143", "144", "145", "146", "147",
                 "148", "149", "150", "151", "152", "153", "154", "155",
                 "156", "157", "158", "159", "160", "161", "162", "163",
                 "164", "165", "166", "167", "168", "169", "170", "171",
                 "172", "174", "175", "176", "177", "178", "179", "180",
                 "181", "182", "183", "184", "185", "186", "187", "188",
                 "189", "190", "191", "192", "193", "194", "195", "200",
                 "201", "202", "203", "204", "205", "206", "207", "208", "238.6"],
        "icd10": ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07",
                  "C08", "C09", "C10", "C11", "C12", "C13", "C14", "C15",
                  "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23",
                  "C24", "C25", "C26", "C30", "C31", "C32", "C33", "C34",
                  "C37", "C38", "C39", "C40", "C41", "C43", "C45", "C46",
                  "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54",
                  "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63",
                  "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71",
                  "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83",
                  "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94",
                  "C95", "C96", "C97"],
        "weight": 2
    },
    "moderate_severe_liver_disease": {
        "icd9": ["456.0", "456.1", "456.2", "572.2", "572.3", "572.4",
                 "572.5", "572.6", "572.7", "572.8"],
        "icd10": ["I85.0", "I85.9", "I86.4", "I98.2", "K70.4", "K71.1",
                  "K72.1", "K72.9", "K76.5", "K76.6", "K76.7"],
        "weight": 3
    },
    "metastatic_solid_tumor": {
        "icd9": ["196", "197", "198", "199.0", "199.1"],
        "icd10": ["C77", "C78", "C79", "C80"],
        "weight": 6
    },
    "aids_hiv": {
        "icd9": ["042", "043", "044"],
        "icd10": ["B20", "B21", "B22", "B24"],
        "weight": 6
    },
}


def load_diagnoses(mimic_path: Path) -> pd.DataFrame:
    """Load diagnosis codes from MIMIC-IV."""
    diag_path = mimic_path / "mimiciv/3.1/hosp/diagnoses_icd.csv.gz"
    logger.info(f"Loading diagnoses from {diag_path}")
    
    df = pd.read_csv(
        diag_path,
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": int}
    )
    
    logger.info(f"Loaded {len(df):,} diagnosis records")
    return df


def load_ed_stays(mimic_path: Path) -> pd.DataFrame:
    """Load ED stays from MIMIC-IV-ED."""
    ed_path = mimic_path / "mimic-iv-ed/2.2/ed/edstays.csv.gz"
    
    if not ed_path.exists():
        logger.warning(f"ED stays file not found at {ed_path}")
        return pd.DataFrame()
    
    logger.info(f"Loading ED stays from {ed_path}")
    df = pd.read_csv(
        ed_path,
        usecols=["subject_id", "stay_id", "intime"],
        parse_dates=["intime"]
    )
    
    logger.info(f"Loaded {len(df):,} ED stays")
    return df


def code_matches(icd_code: str, pattern: str) -> bool:
    """Check if ICD code matches a pattern (prefix matching)."""
    # Clean codes - remove dots for comparison
    clean_code = icd_code.replace(".", "").upper()
    clean_pattern = pattern.replace(".", "").upper()
    
    # Prefix match
    return clean_code.startswith(clean_pattern)


def get_comorbidities_for_admission(
    diagnoses: pd.DataFrame,
    hadm_id: int
) -> Dict[str, bool]:
    """
    Get comorbidity flags for a single admission.
    
    Args:
        diagnoses: All diagnosis records
        hadm_id: Hospital admission ID
    
    Returns:
        Dictionary of comorbidity_name -> bool
    """
    adm_diag = diagnoses[diagnoses["hadm_id"] == hadm_id]
    
    comorbidities = {}
    
    for condition, info in CHARLSON_CODES.items():
        has_condition = False
        
        for _, row in adm_diag.iterrows():
            icd_code = str(row["icd_code"])
            version = row["icd_version"]
            
            # Check appropriate code list based on version
            if version == 9:
                patterns = info["icd9"]
            else:
                patterns = info["icd10"]
            
            for pattern in patterns:
                if code_matches(icd_code, pattern):
                    has_condition = True
                    break
            
            if has_condition:
                break
        
        comorbidities[condition] = has_condition
    
    return comorbidities


def compute_charlson_index(comorbidities: Dict[str, bool]) -> int:
    """
    Compute Charlson Comorbidity Index from comorbidity flags.
    
    Note: Diabetes with complications supersedes without complications.
    Moderate/severe liver disease supersedes mild liver disease.
    Metastatic tumor supersedes malignancy.
    
    Args:
        comorbidities: Dictionary of comorbidity flags
    
    Returns:
        Charlson Comorbidity Index score
    """
    score = 0
    
    for condition, has_condition in comorbidities.items():
        if not has_condition:
            continue
        
        # Handle hierarchical conditions
        if condition == "diabetes_without_complications":
            # Only count if no diabetes with complications
            if not comorbidities.get("diabetes_with_complications", False):
                score += CHARLSON_CODES[condition]["weight"]
        elif condition == "mild_liver_disease":
            # Only count if no moderate/severe liver disease
            if not comorbidities.get("moderate_severe_liver_disease", False):
                score += CHARLSON_CODES[condition]["weight"]
        elif condition == "malignancy":
            # Only count if no metastatic tumor
            if not comorbidities.get("metastatic_solid_tumor", False):
                score += CHARLSON_CODES[condition]["weight"]
        else:
            score += CHARLSON_CODES[condition]["weight"]
    
    return score


def compute_lace_score(
    los_days: float,
    is_emergency: bool,
    charlson_index: int,
    ed_visits_6mo: int
) -> Tuple[int, Dict[str, int]]:
    """
    Compute LACE score for readmission risk.
    
    LACE = Length of stay + Acuity + Comorbidity + ED visits
    
    Args:
        los_days: Hospital length of stay in days
        is_emergency: Whether admission was emergency/urgent
        charlson_index: Charlson Comorbidity Index
        ed_visits_6mo: ED visits in past 6 months
    
    Returns:
        Tuple of (total_score, component_scores)
    """
    # L: Length of stay
    if los_days < 1:
        l_score = 0
    elif los_days == 1:
        l_score = 1
    elif los_days == 2:
        l_score = 2
    elif los_days == 3:
        l_score = 3
    elif los_days >= 4 and los_days <= 6:
        l_score = 4
    elif los_days >= 7 and los_days <= 13:
        l_score = 5
    else:  # 14+
        l_score = 7
    
    # A: Acuity of admission
    a_score = 3 if is_emergency else 0
    
    # C: Charlson Comorbidity Index (capped at 5)
    c_score = min(charlson_index, 5)
    
    # E: ED visits in 6 months (capped at 4)
    if ed_visits_6mo == 0:
        e_score = 0
    elif ed_visits_6mo == 1:
        e_score = 1
    elif ed_visits_6mo == 2:
        e_score = 2
    elif ed_visits_6mo == 3:
        e_score = 3
    else:  # 4+
        e_score = 4
    
    total = l_score + a_score + c_score + e_score
    
    components = {
        "lace_l_score": l_score,
        "lace_a_score": a_score,
        "lace_c_score": c_score,
        "lace_e_score": e_score,
    }
    
    return total, components


def compute_risk_scores(
    cohort: pd.DataFrame,
    diagnoses: pd.DataFrame,
    ed_stays: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute risk scores for each admission in cohort.
    
    Args:
        cohort: DataFrame with cohort admissions
        diagnoses: All diagnosis records
        ed_stays: ED stay records
    
    Returns:
        DataFrame with risk scores and comorbidity flags
    """
    logger.info("Computing risk scores...")
    
    # Ensure datetime types
    cohort = cohort.copy()
    cohort["admittime"] = pd.to_datetime(cohort["admittime"])
    
    # Get cohort hadm_ids for filtering
    cohort_hadm_ids = set(cohort["hadm_id"].unique())
    
    # Filter diagnoses to cohort
    diag = diagnoses[diagnoses["hadm_id"].isin(cohort_hadm_ids)].copy()
    logger.info(f"Filtered to {len(diag):,} diagnoses for cohort")
    
    results = []
    
    for idx, row in cohort.iterrows():
        hadm_id = row["hadm_id"]
        subject_id = row["subject_id"]
        admittime = row["admittime"]
        los_days = row.get("length_of_stay_days", 7)  # Default to 7 if not present
        admission_type = row.get("admission_type", "")
        
        features = {"hadm_id": hadm_id}
        
        # Get comorbidities
        comorbidities = get_comorbidities_for_admission(diag, hadm_id)
        
        # Add individual comorbidity flags
        for condition, has_condition in comorbidities.items():
            features[f"cci_{condition}"] = 1 if has_condition else 0
        
        # Compute Charlson Index
        charlson_index = compute_charlson_index(comorbidities)
        features["charlson_comorbidity_index"] = charlson_index
        
        # ED visits in past 6 months
        ed_visits_6mo = 0
        if len(ed_stays) > 0:
            six_months_ago = admittime - pd.Timedelta(days=180)
            patient_ed = ed_stays[
                (ed_stays["subject_id"] == subject_id) &
                (ed_stays["intime"] >= six_months_ago) &
                (ed_stays["intime"] < admittime)
            ]
            ed_visits_6mo = len(patient_ed)
        
        # Determine if emergency admission
        is_emergency = any(x in str(admission_type).upper() for x in ["EMER", "URGENT"])
        
        # Compute LACE score
        lace_total, lace_components = compute_lace_score(
            los_days, is_emergency, charlson_index, ed_visits_6mo
        )
        
        features["lace_score"] = lace_total
        features.update(lace_components)
        
        # Additional derived features
        features["n_comorbidities"] = sum(
            1 for v in comorbidities.values() if v
        )
        features["has_diabetes"] = 1 if (
            comorbidities.get("diabetes_without_complications", False) or
            comorbidities.get("diabetes_with_complications", False)
        ) else 0
        features["has_cancer"] = 1 if (
            comorbidities.get("malignancy", False) or
            comorbidities.get("metastatic_solid_tumor", False)
        ) else 0
        features["has_liver_disease"] = 1 if (
            comorbidities.get("mild_liver_disease", False) or
            comorbidities.get("moderate_severe_liver_disease", False)
        ) else 0
        
        results.append(features)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"Processed {idx + 1:,}/{len(cohort):,} admissions")
    
    result_df = pd.DataFrame(results)
    logger.info(f"Computed features for {len(result_df):,} admissions")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Extract clinical risk scores")
    parser.add_argument(
        "--cohort_path",
        type=str,
        default="data/interim/readmit_analysis/long_los_cohort.csv",
        help="Path to cohort CSV"
    )
    parser.add_argument(
        "--mimic_path",
        type=str,
        default="physionet.org/files",
        help="Path to MIMIC-IV data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/interim/ehr_long_los/risk_scores.csv",
        help="Output path for features"
    )
    args = parser.parse_args()
    
    # Load cohort
    logger.info(f"Loading cohort from {args.cohort_path}")
    cohort = pd.read_csv(args.cohort_path)
    
    # Filter to cardiorenal long LOS if flag exists
    if "is_cardiorenal_long" in cohort.columns:
        cohort = cohort[cohort["is_cardiorenal_long"] == True].copy()
        logger.info(f"Filtered to {len(cohort):,} cardiorenal long LOS admissions")
    
    # Load data
    mimic_path = Path(args.mimic_path)
    diagnoses = load_diagnoses(mimic_path)
    ed_stays = load_ed_stays(mimic_path)
    
    # Compute features
    features = compute_risk_scores(cohort, diagnoses, ed_stays)
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")
    
    # Print summary
    logger.info("\n=== Risk Score Summary ===")
    logger.info(f"LACE score: mean={features['lace_score'].mean():.2f}, "
                f"median={features['lace_score'].median():.1f}, "
                f"range=[{features['lace_score'].min()}, {features['lace_score'].max()}]")
    logger.info(f"Charlson Index: mean={features['charlson_comorbidity_index'].mean():.2f}, "
                f"median={features['charlson_comorbidity_index'].median():.1f}")
    logger.info(f"Number of comorbidities: mean={features['n_comorbidities'].mean():.2f}")
    
    # Top comorbidities
    logger.info("\n=== Top Comorbidities ===")
    cci_cols = [c for c in features.columns if c.startswith("cci_")]
    prevalence = features[cci_cols].mean().sort_values(ascending=False)
    for col in prevalence.head(10).index:
        logger.info(f"{col.replace('cci_', '')}: {prevalence[col]*100:.1f}%")


if __name__ == "__main__":
    main()
