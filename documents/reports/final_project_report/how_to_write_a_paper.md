## 1. Overall Purpose & Mindset

**Goal of the paper**

* Clearly communicate to your **NLP colleagues**:

  * Your **idea** and **intuition**
  * The **experiments** you designed and ran
  * The **results** you got
  * **Why** you got those results (what worked / didn’t and why) 

Think of it as telling a **coherent story** about your research, not a lab notebook dump.

**Audience**

* Other people who know NLP, so you **don’t** need to explain basics like “what is a neural network” or “what is tokenization.”
* But they **don’t** know your specific project, dataset, or tricks—those you must explain.

**Story, not chronology**

Bad: “We did A, then B, then C, then D…”
Good: “We had this **problem**, we had an **intuition**, we designed **experiments** to test it, and here is **what happened and why**.”

Also: you only have **6 pages max for the main paper** (in this class). Use those pages for the *core story*; push extra details to the appendix. 

---

## 2. Standard Paper Structure (Section by Section)

The instructor gives a standard structure you should follow: 

1. Abstract
2. Introduction
3. Background (Related Work)
4. Methods (Methodology)
5. Results & Discussion
6. Conclusion
7. References
8. Author Contributions (if team)
9. Appendix (optional but recommended)

Below is what goes into each, plus suggested sizes and guiding questions.

---

### 2.1 Abstract (3–5 sentences, write it last)

**Size:** 3–5 sentences. Short and punchy. 

**Content:**

1. **Problem** – What problem are you tackling?
2. **Approach** – What did you do to solve it (at a high level)?
3. **Results** – What did you find (e.g., “Our method improves F1 by X over baseline”)?

It’s a **hook**: someone reads just the abstract and decides whether to continue. For your project:

> Write it *after* you know your full story and results.

---

### 2.2 Introduction (~1 page)

**Size:** ~1 page. 

**Focus:** Motivation + big picture story.

**Questions to answer (in order):**

1. **What is the problem you are trying to solve?**

   * Define the task clearly (e.g., “detecting 30-day readmission from discharge notes”).
2. **Why is it important?**

   * Why should an NLP person care? Real-world impact, difficulty, novelty.
3. **Why can’t it be solved without your approach?**

   * What are existing approaches missing?
   * What are the limitations (shortcomings) of prior work?
4. **What are your contributions?**

   * Example phrasing:

     * “We propose a lightweight classifier that …”
     * “We create a new dataset of …”
     * “We show that [simple method] is competitive with [complex baseline] on …”
   * Keep contributions honest but clear.

You can literally have a short “**Our contributions are:**” bullet list if you want, but the instructor says this is actually the *least* important part of the intro; the main hook is the **problem + why it matters + your idea**.

---

### 2.3 Background / Related Work (~½ page)

**Size:** ~½ page. 

**Focus:** Show you know the literature and where you fit.

**Questions to answer:**

1. **What have others done on this or similar tasks?**
2. **How were they successful?**
3. **How will you go beyond their success / differ from them?**

**Concrete requirements (for class):**

* At least **4 references**. 
* For each reference: **2–3 sentences**:

  * What they did
  * Why it matters for your problem
  * How your work differs (if relevant)

Think: “These are the shoulders we stand on.”

---

### 2.4 Methods / Methodology (2–3 pages)

This is one of the **most important sections**. 

**Size:** 2–3 pages.

**Focus:** Design and implementation of your approach and experiments.

**Questions to answer:**

1. **What is your proposed approach?**

   * Describe your model / pipeline / framework.
   * Use **one or two concrete examples** to illustrate the task and how your system handles it.

2. **What is the intuition behind your approach?**

   * Why should this method help with the problem? (e.g., “We hypothesize that pretraining on clinical notes will help because…”)

3. **How are you solving the problem, step by step?**

   * Data & preprocessing:

     * Datasets used, splits, any filtering.
   * Model details:

     * Architecture, variant (e.g., finetuned BERT vs. logistic regression).
   * Training procedure:

     * Hyperparameters, number of epochs, optimizer, etc. (high level; extreme detail can go in appendix).

4. **What are your experiments / experimental design?**

   * Describe each experiment in a way that shows **why you chose it**.
   * For example:

     * Baseline vs. your method
     * Ablation (remove a feature / component)
     * Different data sizes
   * Explain how this design tests your hypothesis.

5. **How are you measuring success?**

   * What metrics? (Accuracy, F1, macro-F1, AUROC, etc.)
   * Why those metrics are appropriate for your task.

You can think of this section as:

> “If another NLP student wanted to reproduce my work, could they do it from this description?”

---

### 2.5 Results & Discussion (1–2 pages)

**Size:** 1–2 pages. 

**Focus:** What happened + why it happened.

You need **both**:

1. **Results (How well)**

   * Compare your model vs. baseline(s).
   * Use at least **one main table** summarizing experiments:

     * Typically: **Rows = datasets / data variants**, **Columns = experiments / models**. 
   * You don’t need to include every single run—only those that are **material to your story**.

2. **Discussion / Analysis (Why)**

   * Why did your model perform this way?
   * What patterns do you see?
   * Where does the model do well? Where does it fail?
   * Are there interesting trends across datasets, hyperparameters, or data sizes?

**Error analysis & probing**

* Look at errors your model makes:

  * Are there systematic patterns? (e.g., fails on long sentences, negation, rare classes)
* Sometimes design **extra mini-experiments** (or hand-crafted examples) to test your explanation:

  * “We suspect the model relies heavily on X; we create examples emphasizing X and see performance change.”

**Tables as story anchors**

* Your main table(s) act as **anchor points** for your narrative:

  * You refer to them in the text:

    * “As shown in Table 1, our method improves macro-F1 by 5 points over the baseline on Dataset A.”
  * They help keep the section focused and compact.

---

### 2.6 Conclusion (3–5 sentences)

**Size:** 3–5 sentences. 

**Purpose:** Bookend to your introduction.

**Questions to answer:**

1. **What problem did you tackle?**
2. **Did you solve it (fully or partially)?**
3. **What did you find?**

   * Restate key takeaway(s) succinctly.
4. **Future work (optional)**

   * One or two realistic directions you would pursue with more time / resources.

Don’t introduce brand-new results here. Just synthesize.

---

### 2.7 References

**Goal:** Show you actually consulted the literature. 

* Use a consistent citation style (e.g., ACL style).
* Put all papers you cited in Background and possibly others you referenced in Methods / Discussion.
* For this class: **≥ 4 references is the minimum**, but more is fine and usually better.

---

### 2.8 Author Contributions (for 2–3 person teams)

If you’re in a group of **2 or 3**, you **must** include an “Authors’ Contributions” section. 

Example template (adapted from the slide):

> **Authors’ Contributions**
> A.B., C.D., and E.F. jointly conceptualized the project.
> A.B. implemented the baseline model and conducted initial data analysis.
> C.D. implemented the [Model 1] experiments.
> E.F. implemented the [Model 2] experiments and error analysis.
> A.B. drafted the manuscript; C.D. and E.F. revised and edited it.
> All authors read and approved the final manuscript.

---

### 2.9 Appendix

Because of the **4–6 page limit** on the main body, use the **appendix** for extra details. 

Things you can put there:

* Full results tables (all hyperparameter sweeps, extra experiments)
* Detailed model / training settings
* Additional example predictions and error analyses
* Extra data samples or annotation guidelines

In the main text, you can write things like:

> “We tried additional learning rates; see Appendix A for details.”

This is exactly what many real papers do (e.g., T5 paper: ~10-page main text, huge appendix).

---

## 3. How to Actually *Write* It (Process Tips)

The instructor also gives process advice:

1. **Don’t think “I must write a whole paper.”**
   Think: “I’m going to write **this section** now.”

2. **Write messy first, revise later.**

   * For many people, the key is to **get something on the page** and then iterate.
   * Your final introduction is almost never the one you drafted first.

3. **Use the structure as a checklist.**

   * For each section, ask:

     * Did I answer all the guiding questions?
     * Is the size roughly right?

4. **Write the abstract last.**

   * Once you really know what your story and main results are.

5. **Formatting**

   * For “ACL-style” look and feel:

     * There are ACL templates (LaTeX and Word) linked from the project page.
     * You can use **Overleaf** with the ACL LaTeX template if you’re comfortable.
     * Or just use Word; formatting is *not* the main grading criterion.

---

## 4. Publication vs. Class Project

From the Q&A at the end:

* Your **class project** is not required to be “conference-worthy.”
* Typical conference papers:

  * Longer than 6 pages
  * Require **more extensive literature review**
  * Usually based on work done over **months to a year**, not a single semester.
* If you *do* want to pursue publication:

  * You’d likely need to run **more experiments** and deepen the analysis.
  * You’d check the **call for papers (CFP)** of the target conference for specific requirements.
  * Instructor is open to talking about this *after* seeing the project.

---

## 5. Concrete Checklist for Your Report

You can literally use this as a writing checklist:

### Abstract (3–5 sentences)

* [ ] 1–2 sentences: What is the problem?
* [ ] 1–2 sentences: What method / approach did you use?
* [ ] 1 sentence: What are your main results?

### Introduction (~1 page)

* [ ] Clearly state the problem and why it matters.
* [ ] Explain why existing methods aren’t enough.
* [ ] Introduce your high-level idea / approach.
* [ ] Briefly highlight your contributions (bullet list optional).

### Background (~½ page)

* [ ] Include at least 4 references.
* [ ] For each: 2–3 sentences on what they did and how it relates.
* [ ] End by explaining how your work builds on or differs from these.

### Methods (2–3 pages)

* [ ] Describe the task with 1–2 example inputs/outputs.
* [ ] Explain your intuition for the approach.
* [ ] Describe datasets, preprocessing, and splits.
* [ ] Describe model(s) and training setup.
* [ ] Clearly specify your experiments (what varies, why).
* [ ] Clearly define evaluation metrics and why they’re appropriate.

### Results & Discussion (1–2 pages)

* [ ] Include at least one main table summarizing model vs. baseline, etc.
* [ ] Describe *what* the results are (how well).
* [ ] Analyze *why* they look this way (patterns, data regimes, ablations).
* [ ] Include some error analysis / qualitative observations.
* [ ] If you claim something, back it up with numbers or examples.

### Conclusion (3–5 sentences)

* [ ] Restate problem and your approach.
* [ ] Summarize main findings.
* [ ] Briefly mention limitations and/or future work (optional).

### Contributions (if team)

* [ ] Add a short section listing what each person did.

### References

* [ ] All cited work is included and formatted consistently.

### Appendix (optional but useful)

* [ ] Extra tables, plots, hyperparameter sweeps, sample outputs, etc., moved out of main text.
* [ ] Main text refers to appendix where relevant (“see Appendix A”).

