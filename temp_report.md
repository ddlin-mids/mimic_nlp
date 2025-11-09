# Notebook Overviews

## notebooks/notebooks_dc/01_label_construction.ipynb
**Goal.** Build the long-stay (>15 day) cohort and attach a 30-day readmission label so later notebooks can join notes and imaging. Everything runs out of Google Colab and points to `/content/drive/MyDrive/MIDS/w266/Final Project/` for both raw tables (`data/`) and outputs (`output/`).

**Inputs.** `admissions.csv` (546k encounters) and `patients.csv` from the MIMIC-IV core schema are loaded, merged on `subject_id`, and expanded with `anchor_age`, `anchor_year`, and `gender` to infer each admission's age. Datetime parsing covers `admittime/dischtime/deathtime` plus ED timestamps, and the notebook does quick sanity checks on schema, missing values, and frequency tables for admission/discharge types.

**Label construction workflow.**
- Compute `los_days = dischtime - admittime` and descriptive stats, then drop rows with missing `age_at_admit`.
- Apply exclusion criteria sequentially: remove in-hospital deaths (`hospital_expire_flag == 1`), discharges to `DIED/HOSPICE/AGAINST ADVICE`, and finally enforce `los_days >= 15` to focus on complex long-stay cases.
- Sort all remaining admissions per patient (`all_admissions_sorted`) and iterate through the filtered “index admissions,” scanning forward in each patient’s history to find the next admission that starts within 30 days. The loop populates `readmitted_30day`, `days_to_readmission`, and `readmission_hadm_id` and prints running progress.
- Summaries include class balance, a histogram of days-to-readmission saved as `output/step1_days_to_readmission.png`, and age/LOS distributions after filtering (expected readmission rate ≈22–25%).

**Outputs.** Columns are renamed to the partner conventions (`readmitted_within_window`, `readmission_gap_in_days`, `length_of_stay_days`), and the curated table is written to `output/labeled_admissions_30day_readmission.csv`. The final log also surfaces the class ratio, LOS stats, and file path so downstream notebooks know where to pick up the cohort.

**Notable considerations.** Processing is entirely pandas-based, so the O(N) loop over ~50–60k admissions could become a bottleneck if the cohort grows; there are no intermediate persistence layers besides the final CSV and PNG. The notebook assumes raw CSVs are already synced to Drive and read/write permissions are available.

## notebooks/notebooks_dc/02_link_notes_to_admissions.ipynb
**Goal.** Start from `labeled_admissions_30day_readmission.csv`, attach discharge summaries and chest X-ray reports, and keep only encounters that have both note modalities for multimodal modeling.

**Data loading & profiling.** The notebook reloads the labeled cohort (still using the `readmitted_30day` column name), `discharge.csv` (331k summaries), `radiology.csv` (~2.3M reports), and `radiology_detail.csv`. It prints schema/size info, inspects radiology categories, and peeks at text samples to understand how to filter chest studies.

**Discharge note linkage.** Admissions are inner-joined with discharge notes on `hadm_id`, keeping charttime/note_id metadata and renaming the text column to `discharge_text`. Coverage after this step is 311,459 admissions (~60% of the labeled set) with a slightly higher readmission rate.

**Chest X-ray selection.**
- Text-based filtering looks for “chest” keywords while excluding CT phrases, then the notebook reduces to one report per admission by sorting on `charttime` and keeping the last (closest to discharge). Diagnostics show there are ≈561k raw CXR reports spanning ~180k hadm_ids.
- Additional exploration of `radiology_detail` is included in case text lives there, but the current run relies on the main `radiology.csv` text field.

**Merging both modalities.** Only admissions with both a discharge summary and a CXR are retained (140,685 rows, 27.1% of the start). The notebook reports:
- Coverage percentages relative to the original cohort and the discharge-only subset.
- Label distribution (109,644 negative / 31,041 positive; 22.06% readmitted).
- Text length diagnostics (mean discharge note ≈11.6k characters, radiology ≈584 characters, combined ≈12.2k). Estimated tokens use chars/4 to confirm that 99.7% of cases stay under BioClinical-ModernBERT’s 8,192 token limit; only 354 encounters require truncation.
- Histograms saved to `output/note_length_distributions.png`.

**Temporal validation.** Radiology chart times are compared to discharge times; 99.9% occur before discharge and any post-discharge reports (168 cases) are filtered out to maintain causal ordering.

**Outputs.** The merged dataset (with both note texts, timestamps, note_ids, demographics, and the 30-day label) is saved as `output/labeled_admissions_with_discharge_and_radiology.csv` (~1.6 GB, 18 columns). The notebook closes with a consolidated summary table and next-step pointer toward encoding.

**Caveats.** All filtering hinges on keyword heuristics; no structured “cxr” flag is used. The notebook still references the pre-renaming label column (`readmitted_30day`), so rerunning after Notebook 1’s renames requires either reverting the column names or using the copied variant below.

## notebooks/notebooks_dc/Copy of 02_link_notes_to_admissions.ipynb
**Purpose.** Functionally identical to Notebook 2, but updated to align with the renamed columns exported by Notebook 1 (`readmitted_within_window`, `length_of_stay_days`, etc.). It serves as a rerunnable template once the upstream naming convention change landed.

**Key differences vs the original 02 notebook.**
- Adds a defensive check that the expected renamed columns exist and prints age/LOS summaries from them before any joins.
- All downstream stats (rate calculations, summaries) reference `readmitted_within_window` instead of `readmitted_30day` to avoid KeyErrors.
- Otherwise, the data loading, chest X-ray filtering, “keep last report per admission” logic, note-length analysis, temporal validation, and CSV export steps mirror Notebook 2, producing the same `labeled_admissions_with_discharge_and_radiology.csv` artifact in Drive.

**Implication.** Treat this as the maintained version if you plan to re-run the cohort linkage; it keeps the narrative/plots identical while preventing mismatches with the standardized schema.

## notebooks/notebooks_dc/04B_encode_notes_bioclinical_modernbert.ipynb
**Goal.** Consume the multimodal cohort from Notebook 2, encode discharge and radiology texts with `thomas-sounack/bioclinical-modernbert-base`, and persist embeddings for modeling.

**Setup.** Runs in Colab with GPU detection logic (expects an A100). It mounts Drive, points `OUTPUT_DIR` to the same project folder, and creates an `output/embeddings/` directory. Dependencies include PyTorch, Hugging Face Transformers, tqdm, numpy/pandas, and matplotlib (used for diagnostics).

**Model loading & smoke test.**
- Downloads tokenizer/model with an 8,192-token context window, pushes the model to CUDA, and disables grad tracking.
- Tokenizes the first ~1,000 characters of a discharge note to verify max-length handling, prints CLS embedding norms, and confirms inference works before the full batch run.

**Encoding pipeline.**
- Adds quick token-length estimates (chars/4) for both note types to reconfirm that only 0.3% of discharges exceed the window; radiology reports are far shorter.
- Defines `encode_texts(...)`, which batches texts (default batch_size=16), tokenizes with truncation/padding to 8,192 tokens, runs the model, extracts the CLS vector (`last_hidden_state[:, 0, :]`), and writes each batch to a numpy array while freeing GPU memory via `del` and `torch.cuda.empty_cache()`.
- Runs the function twice: once for all discharge notes (estimated 1–2 h on an A100) and once for radiology notes (~30–60 min). Shapes, means, and std devs are logged for sanity checking.

**Persistence & QC.** Both embedding matrices plus their `hadm_id` mapping are saved to compressed `.npz` files:
- `embeddings/discharge_embeddings_140k.npz`
- `embeddings/radiology_embeddings_140k.npz`
- `embeddings/embedding_hadm_id_mapping_140k.csv`
The notebook reloads the `.npz` files to confirm shapes match expectations and plots log-scale histograms of embedding values for each modality.

**Notes.** There is no downstream training in this notebook—its remit ends once embeddings are serialized. Runtime expectations (~2–3 hours total) and hardware requirements are clearly stated so planners can book GPU time accordingly.
