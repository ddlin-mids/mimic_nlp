# MIMIC-IV and Related Datasets: Summary

This document summarizes the MIMIC-IV ecosystem of de-identified clinical datasets from Beth Israel Deaconess Medical Center (BIDMC), hosted on PhysioNet. MIMIC-IV focuses on ICU and hospital data (2008–2022), emphasizing privacy via HIPAA-compliant de-identification. All datasets require credentialed access (DUA, CITI training). Key guidelines: Treat derived data sensitively; use "Ext" in names (e.g., MIMIC-IV-Ext); cite sources and PhysioNet.

## Core Citations
- **MIMIC-IV**: Johnson et al. (2024). PhysioNet. DOI: 10.13026/kpb9-mt58. Original: Johnson et al. (2023). Sci Data. DOI: 10.1038/s41597-022-01899-x.
- **PhysioNet Standard**: Goldberger et al. (2000). Circulation. RRID: SCR_007345.

## 1. MIMIC-IV (v3.1, Oct 2024)
### Overview
Large de-identified EHR dataset for >364K patients (223K hospitalized, 141K ED-only). Supports epidemiology, predictive modeling. Modules: **hosp** (hospital-wide EHR, 546K admissions) and **icu** (MetaVision ICU system, 94K stays for 65K patients).

### Key Features
- **De-identification**: Random patient IDs; dates shifted to 2100–2200 (patient-specific, preserves internal timelines); PHI removed/replaced (e.g., ___ in text).
- **Anchor Info**: `anchor_year` (de-ID year), `anchor_year_group` (e.g., 2008–2010), `anchor_age` (age at anchor; >89 grouped as 91).
- **Death Linkage**: `dod` in `patients` (hospital/state records; censored >1 year post-discharge).
- **hosp Module Tables**: patients, admissions, transfers, labevents/d_labitems, microbiologyevents/d_micro, poe/poe_detail, emar/emar_detail, prescriptions/pharmacy, diagnoses_icd/d_icd_diagnoses, procedures_icd/d_icd_procedures, hcpcsevents/d_hcpcs, drgcodes, omr, services, provider.
- **icu Module Tables**: icustays, d_items, inputevents, ingredientevents, outputevents, procedureevents, datetimeevents, chartevents, caregiver.
- **Linkages**: To MIMIC-IV-Note (notes), MIMIC-IV-ED (ED data), MIMIC-CXR (x-rays). Use `subject_id` + date alignment.

### Access & Availability
- BigQuery: `mimiciv_v3_1_hosp`/`mimiciv_v3_1_icu` (request via PhysioNet). v2.2 on `mimiciv_v2_2_*`; legacy `mimiciv_*` to be replaced Nov 25, 2024.
- Size: ~4.7 TB uncompressed (full MIMIC-CXR included in ecosystem).

### Release Notes
| Version | Date | Key Changes |
|---------|------|-------------|
| v3.1 | Oct 2024 | Fixed lab item IDs (consistent w/ v2.2); removed 2 invalid subjects; updated tables: d_labitems, diagnoses_icd, etc. |
| v3.0 | Jul 2024 | Added 2020–2022 data; out-of-hospital mortality (1-yr post-discharge); improved language/insurance; +65K patients, +115K admissions, +21K ICU stays. |
| v2.2 | Jan 2023 | Added provider IDs (caregiver/provider tables); imputed hadm_id in emar; removed test set (~5% rows). |
| v2.1 | Nov 2022 | Removed test set subjects. |
| v2.0 | Jun 2022 | Removed neonates/core module; added out-of-hospital death, omr (e.g., height/weight/eGFR); ingredientevents; fixed bugs (e.g., edregtime). |
| v1.0 | Mar 2021 | Initial; fixed hadm_id/dod; regenerated IDs. |

**Usage Tips**: Raw clinical data—expect implausibles/missing values. Docs: MIMIC-IV site; MIMIC Code Repo (GitHub).

## 2. MIMIC-IV-Note (v2.2, Jan 2023)
### Overview
De-identified free-text notes from MIMIC-IV patients. 331K discharge summaries (146K patients) + 2.3M radiology reports (237K patients). Supports NLP (e.g., entity recognition, de-ID).

### Key Features
- **De-ID**: Rule-based + neural net; PHI → ___; sensitivity 99.9% for radiology.
- **Tables**: discharge/discharge_detail (summaries + authors); radiology/radiology_detail (reports + CPT/exam/addendums).
- **Linkage**: Via `subject_id` + dates to MIMIC-IV.

### Release Notes
- v2.2: Metadata updates only.
- v2.1: Initial public release.

**Size**: 1.8 GB. DOI: 10.13026/1n74-ne17.

## 3. MIMIC-IV-Ext-22MCTS (v1.0.0, Sep 2025)
### Overview
Derived from MIMIC-IV-Note: 22.6M temporal clinical events + timestamps from 267K discharge summaries. Uses Llama-3.1-8B for extraction (events as text spans, relative hours from admission t=0). For risk prediction, trajectory analysis.

### Key Features
- **Events**: Avg 84/summary (1–244); 37% historical (negative time), 51% during admission, 12% post-discharge. Avg 3 tokens/event.
- **Table Columns**: hadm_id, Event (text), Time (hours), Time_bin (discrete: e.g., Bin 0: <-60 hrs, Bin 8: >120 hrs).
- **Extraction**: BM25 + BGE embedding retrieval; LLM prompting (91% event concordance w/ human on test cases).
- **Limitations**: Possible hallucinations/imprecision; no gold labels (best for pretraining/weak supervision).

### Usage
- Fine-tune for QA, trial matching (e.g., +10% PubMedQA). Code: GitHub.

**Parent**: MIMIC-IV-Note v2.2. DOI: 10.13026/dkj6-r828.

## 4. MIMIC-IV-ED (v2.2, Jan 2023)
### Overview
~425K ED stays (2011–2019) from BIDMC. Vital signs, triage, meds, diagnoses. Links to MIMIC-IV (pre-ICU data) and MIMIC-CXR.

### Key Features
- **Tables**:
  - edstays: Stays (intime/outtime, gender/race, transport/disposition).
  - diagnosis: ICD-9/10 codes (up to 9/stay).
  - medrecon: Pre-ED meds (name/GSN/NDC + ontology).
  - pyxis: Dispensed meds (name/GSN).
  - triage: Initial vitals (temp/HR/RR/O2/SBP/DBP/pain), acuity (1–5), chiefcomplaint.
  - vitalsign: Ongoing vitals + rhythm/pain.
- **De-ID**: Numeric vitals only; free-text PHI → ___.

### Release Notes
- v2.2: Removed test set (~22K stays).
- v2.0: Added demographics/transport; fixed outtime.
- v1.0: Initial.

**Size**: 116 MB. DOI: 10.13026/5ntk-km72.

## 5. MIMIC-CXR (v2.1.0, Jul 2024)
### Overview
377K chest x-rays (DICOM) + reports for 228K studies (2011–2016). For image NLP, decision support.

### Key Features
- **Structure**: Folders by patient (p10–p19); studies (s50M–s59M) w/ images + .txt reports.
- **Metadata**: cxr-record-list.csv (image-study-patient links); cxr-study-list.csv; cxr-provider-list.csv (ordering/attending/resident IDs, link to MIMIC-IV providers); mimic-cxr-reports.tar.gz.
- **De-ID**: Burned-in text removed (black boxes); OCR + rules.

### Release Notes
- v2.1.0: Added providers.
- v2.0.0: DICOM format; reorganized IDs.

**Size**: 4.7 TB. DOI: 10.13026/4jqj-jw95.

## Ethics & Acknowledgements
- IRB: Waiver of consent (BIDMC #2001P001699).
- Funding: NIH R01EB030362; Philips (CXR).
- Conflicts: None (except Philips for CXR).

For full details/access: physionet.org; MIMIC site (mimic.mit.edu); Code Repo (GitHub/MIT-LCP/mimic-code).

here’s a skimmable, researcher-friendly summary 👇

# MIMIC quick status

* **BigQuery availability:**

  * **MIMIC-IV v3.1** → `mimiciv_v3_1_hosp`, `mimiciv_v3_1_icu`.
  * **MIMIC-IV v2.2** → `mimiciv_v2_2_hosp`, `mimiciv_v2_2_icu`, and legacy `mimiciv_hosp`, `mimiciv_icu`.
  * **Changeover:** on **Nov 25, 2024**, `mimiciv_hosp`/`mimiciv_icu` were replaced with **v3.1**.
* **Access:** request via **PhysioNet** (credentialed, DUA, CITI “Data or Specimens Only” training).

# Using/deriving datasets & models (Guidelines)

* Treat any **derived datasets/models** as **sensitive**; share on **PhysioNet** under the **same agreement** as source.
* If using “MIMIC” in the name, append **“Ext”** (e.g., *MIMIC-IV-Ext-YOUR-DATASET*).
* In PhysioNet submission, set **Parent Projects** appropriately.
* **Cite**:

  * Johnson et al., **MIMIC-IV v3.1** (2024) + **Sci Data** paper (2023).
  * **PhysioNet standard** citation.

# What’s in MIMIC-IV (core facts)

* Scope: **ED** + **ICU** patients at BIDMC; **>65k ICU** patients, **>200k ED** patients.
* **Modules:**

  * **hosp** (EHR-wide: demographics, admissions/transfers, labs, meds, billing, providers, OMR, etc.).
  * **icu** (MetaVision “events” star schema: chartevents, input/output, procedures, etc.).
* **De-ID:** patient-specific date shifts (to 2100–2200), randomized IDs, PHI removed; **within-patient timing preserved**, **across-patient calendar comparison invalid**. **Death dates** captured (censored >1 year post last discharge).
* **Anchor fields:** `anchor_year`, `anchor_year_group`, `anchor_age` (age 90+ → 91).

# Linkable companion projects

* **MIMIC-IV-Note:** de-identified **free-text** (discharge summaries + radiology reports); link by `subject_id`.
* **MIMIC-IV-ED:** ~**425k ED stays** (2011–2019): vitals, triage, meds, diagnoses; linkable via `subject_id/hadm_id/stay_id`.
* **MIMIC-CXR:** **chest X-rays** (DICOM) with reports; provider IDs align with MIMIC-IV ≥ v2.2.

# Version history (high-level)

* **v3.1 (Oct 2024)** – bugfixes & consistency: restored **lab itemIDs** parity with v2.2; removed 2 orphan `subject_id`; updated: `d_labitems`, `labevents`, `microbiologyevents`, `omr`, `transfers`, `icustays`, `diagnoses_icd`, `drgcodes`.
* **v3.0 (Jul 23, 2024)** – **adds 2020–2022**; **out-of-hospital mortality up to 1-year**; counts ↑ (`patients` 364,627; `admissions` 546,028; `icustays` 94,458). Better **language** & **insurance** fields.
* **v2.2 (Jan 2023)** – adds **provider identifiers** (hosp `provider`, icu `caregiver_id`), fills missing `hadm_id` in `emar`.
* **v2.1 (Nov 2022)** – removes held-out **test** subjects/stays.
* **v2.0 (Jun 2022)** – schema simplification (remove `core`), add **OMR**, **out-of-hospital death**, ICU infusion detail, neonatal data removed to separate project.
* **v1.0 (Mar 2021)** – major ICU stay ID fixes; many table cleanups.

# Project spotlights (extras you mentioned)

**MIMIC-IV-Note (v2.2, 2023)**

* **331,794 discharge summaries** (145,915 pts) + **2,321,355 radiology reports** (237,427 pts).
* De-ID via rule-based + neural; PHI → “___”. CSV delivery + `_detail` EAV tables.

**MIMIC-IV-Ext-22MCTS (v1.0.0, Sept 29, 2025)**

* **22,588,586** event–timestamp pairs from **267,284** discharge summaries (from MIMIC-IV-Note).
* Events extracted with retrieval + **Llama-3.1-8B**; time is **relative to admission (t=0)**; also **binned (0–8)**.
* Intended for **temporal modeling** (risk prediction, next-event, timeline reasoning).
* **Caveats:** LLM-generated → possible hallucinations/misaligned times; good for **pretraining/weak supervision**, not gold-standard evaluation.

**MIMIC-IV-ED (v2.2, 2023)**

* Star schema around `edstays`; tables: `diagnosis`, `medrecon`, `pyxis`, `triage`, `vitalsign`.
* De-ID consistent with MIMIC-IV; linkable via IDs.

**MIMIC-CXR (v2.1.0, Jul 23, 2024)**

* **377,110 images** for **227,835 studies**; DICOM + reports; de-ID includes pixel PHI removal; provider IDs align with MIMIC-IV.

# Practical tips for you

* **Pick the right schema:** prefer `mimiciv_v3_1_*` on BigQuery; legacy `mimiciv_*` now points to v3.1 since **Nov 25, 2024**.
* **Timing logic:** always reason **within subject**; use `anchor_*` to align to real-world year-bands.
* **Mortality analyses:** follow-up capped at **1 year** post last discharge.
* **Provider linkage:** use `provider` (hosp) and `caregiver_id` (icu); `provider_id` prefixes differ by table context.
* **When publishing:** include **all three citations** (MIMIC-IV dataset DOI, Sci Data paper, PhysioNet standard).

If you want, I can turn this into a one-pager (Markdown or Google Doc style) with a tiny “Which table do I need?” cheat sheet.
