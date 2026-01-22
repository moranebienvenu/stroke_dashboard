# Stroke Dashboard

The main objective of this app is to allow interactive exploration of PSA patient data derived from T1 maps and language assessments across the acute, subacute, and chronic phases.

This dashboard is designed to visualize and interact with outputs previously generated using [NeuroTmap](https://github.com/Pedro-N-Alves/NeuroT-Map) (Alves, P.N., Nozais, V., Hansen, J.Y. et al., *Neurotransmitters’ white matter mapping unveils the neurochemical fingerprints of stroke*, Nat Commun 16, 2555 (2025), [https://doi.org/10.1038/s41467-025-57680-2](https://doi.org/10.1038/s41467-025-57680-2)).

It allows users to explore the data interactively, modify filters, and examine the relationships between different metrics at the subject or group level for big dataset. It allows add the possibility to run statistical analysis such as GLM, T-test or correlation. 

---

## Input Data Requirements

The dashboard requires the following files:

- One or more `output_les_dis_sub-XXX_ses-VX.csv` files  
- One or more `output_pre_post_synaptic_ratio_sub-XXX_ses-VX.csv` files  
- (Optional) `clinical_data.csv` or `.xlsx` file  

**Important notes:**

- **Subject IDs format:** `sub-<group letter><digits>_ses-V<session number>`  
  Examples: `sub-A01_ses-V1`, `sub-G1234_ses-V2`
- **Group letters list:**  
  - NA: Non-aphasic  
  - A: Aphasic  
  - G: Global  
  - W: Wernicke  
  - B: Broca  
  - C: Conduction  
  - AN: Anomic  
  - TCM: Transcortical Motor  
  - TCS: Transcortical Sensory  
  - TCMix: Transcortical Mixed
- The `clinical_data` file must include a `subject` column matching the filenames exactly.
- Additional columns (optional): `sex`, `timepoint`, `repetition_score`, `comprehension_score`, `naming_score`, `composite_score`, `lesion_volume`.  
- **Lesion volume must be in mm³.**

You can try the example dataset first to use the dashboard without personal datas: [synthetic_dataset.zip](https://github.com/moranebienvenu/stroke_dashboard).

---

## Installation

1. Create a Python environment (e.g., using `conda` or `venv`).
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the dashboard in your terminal:
```bash
python app.py
```

## About

This dashboard is also the data source for a MyST interconnected article, a next-generation open science publication (“wired-article”) soon available on [Evidence](https://evidencepub.io/). Here is a preview link now available of the article: [Sex-specific early neurotransmitter dynamics and post-stroke aphasia recovery](https://preview.neurolibre.org/myst/moranebienvenu/stroke_article/e0ea9d330c2d6aebf81ec6a435108ae4c050a4db/_build/html/index.html).

It exemplifies how interactive dashboards can fully connect datasets, code, and figures within scientific publications, enabling reproducible and transparent neuroscience research. The next step toward truly FAIR neuroscience publications. 