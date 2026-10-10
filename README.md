# RetinaAssist

**An explainable retinal screening research prototype. Not a medical device.**

RetinaAssist explores diabetic retinopathy (DR) screening from retinal fundus images. It combines image-quality checks, PyTorch image classifiers, IDRiD lesion segmentation, lesion-feature fusion experiments, Grad-CAM visualizations, and an illustrative clinic-capacity simulation.

> **Research use only.** RetinaAssist has not been clinically validated. Do not use it to diagnose, screen, make treatment decisions, or rule out DR. Model scores and visualizations are not clinical findings.

## What’s implemented

- APTOS binary DR classifier using an ImageNet-pretrained ResNet-18
- Five-grade DR classification (grades 0–4), with APTOS-only and mixed APTOS/IDRiD training experiments
- IDRiD segmentation model for microaneurysms, haemorrhages, hard exudates, soft exudates, and optic disc
- Image-quality checks, retinal-image preprocessing, and a Streamlit demo
- Grad-CAM explanations and segmentation overlays
- Lesion-feature fusion experiment comparing image-only and fused grading
- Python clinic-capacity scenario simulation
- Evaluation scripts for held-out APTOS data and official IDRiD test splits

The demo presents a **Demo diagnosis** for research demonstration. Grad-CAM highlights influential regions; it does not identify lesions. Segmentation outputs are model estimates, not confirmed findings.

## Reported results

These are results from the current experiments, not estimates of clinical performance.

| Experiment | Evaluation set | Reported results |
|---|---:|---|
| APTOS binary classifier | 550-image stratified held-out split | Accuracy 97.6%, sensitivity 97.1%, specificity 98.2%, ROC-AUC 0.998 |
| APTOS five-grade baseline | 550-image held-out split | Accuracy 79.1%, macro-F1 0.628, quadratic weighted kappa (QWK) 0.874 |
| APTOS lesion-fusion grading | Same 550-image held-out split | Accuracy 80.7%, macro-F1 0.625, QWK 0.865 |
| APTOS-trained five-grade model | IDRiD official grading test split, 103 images | Accuracy 45.6%, macro-F1 0.369, QWK 0.596 |
| IDRiD lesion segmentation | Official segmentation test split, 27 images | Macro Dice 0.619 |

The APTOS binary result comes from a held-out split of labeled training data; it is **not** an evaluation on the competition’s hidden test set. The APTOS grading results show a tradeoff between the baseline and lesion-fusion models; fusion did not improve every metric. IDRiD grading performance is substantially weaker than the internal APTOS result, illustrating the external-domain performance gap.

The capacity simulation is illustrative only. Under its current assumptions, it estimates about 79 clinician reviews per day with AI-assisted triage versus 151 with manual review, and about 3.7 simulated missed cases per day. These are scenario outputs based on limited research metrics, not real-world outcomes, staffing advice, or evidence that clinician review can be safely reduced.

The reported values summarize prior local runs. Checkpoints and generated reports are not bundled, so retraining may produce different metrics. The binary run's detailed metrics are included in `models/best.metrics.json`; the grading and segmentation results above are summarized from local evaluation reports.

## Setup

Use Python 3.11 or a compatible version. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

## Data

RetinaAssist does not distribute datasets. Obtain each dataset from its authorized source and follow its terms.

### APTOS 2019

Download the APTOS 2019 Blindness Detection data from [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data). Place the labeled CSV at `data/labels.csv` and the corresponding images in `data/raw/`. The native `id_code,diagnosis` CSV format is supported. For binary training, grades are mapped to no DR (0) versus suspected DR (1–4); the five-grade experiment retains grades 0–4.

### IDRiD

Download the disease-grading and segmentation data from the authorized [IDRiD source](https://idrid.grand-challenge.org/). Keep the original directory structure expected by the training scripts under `data/external/idrid/`. Missing segmentation annotations are treated as unavailable labels, not as negative masks.

### Messidor-2

No Messidor-2 evaluation result is currently reported. The dataset and its reference labels must be obtained and checked against their applicable terms before an external evaluation can be run.

Do not commit image data, labels, or other restricted dataset files.

## Training and evaluation

### APTOS binary experiment

```bash
python -m src.train \
  --csv data/labels.csv \
  --image-root data/raw \
  --pretrained \
  --epochs 8 \
  --batch-size 16
```

Evaluate the saved binary checkpoint:

```bash
python -m src.evaluate \
  --csv data/labels.csv \
  --image-root data/raw \
  --checkpoint models/best.pt
```

### Five-grade, segmentation, and fusion experiments

The Streamlit demo uses the five-grade APTOS classifier and the IDRiD segmentation model. Train both checkpoints before launching it:

```bash
python -m src.train_grading \
  --csv data/labels.csv \
  --image-root data/raw \
  --pretrained \
  --epochs 20 \
  --batch-size 16

python -m src.train_segmentation \
  --root "data/external/idrid/A. Segmentation" \
  --epochs 30 \
  --batch-size 4
```

To recreate the baseline model's APTOS-validation thresholds used in the demo, run:

```bash
python -m src.calibrate_grade5_thresholds \
  --checkpoint models/aptos_grade5_best.pt \
  --output outputs/aptos_grade5_thresholds.json
```

The demo falls back to 0.5 thresholds if that optional JSON file is absent, and labels the threshold source in its output. Recalibrate after retraining; thresholds belong to the checkpoint used to select them.

The corresponding entry points are:

- `src.train_grading`
- `src.train_mixed_grading`
- `src.train_segmentation`
- `src.evaluate_segmentation`
- `src.train_fusion_grading`
- `src.evaluate_idrid_grade5`

Argparse-based scripts support `python -m MODULE --help`, replacing `MODULE` with the entry point. A few evaluation scripts use fixed paths near the top of the file. Do not use an official test split to select checkpoints or tune thresholds.

### Demo and tests

```bash
streamlit run app.py
python -m pytest -q
```

The demo requires the trained grading and segmentation checkpoints in `models/`. Checkpoint files are not included in the repository. Image datasets, labels, model weights, and generated outputs remain local and are excluded from Git.

## Project layout

```text
app.py                 Streamlit research demo
src/                   Training, evaluation, inference, and simulation code
tests/                 Automated tests
data/                  Locally stored datasets; excluded from Git
models/                Locally stored checkpoints; excluded from Git
outputs/               Generated reports and simulation results; excluded from Git
```

## Limitations

- APTOS and IDRiD differ in image sources and data distributions; internal metrics do not establish external performance.
- The IDRiD grading test set contains 103 images, so its estimates are limited and uncertain.
- The current external evaluation shows a specificity gap; threshold selection on APTOS validation data did not remove the domain gap.
- The segmentation model was trained on a small set of annotated IDRiD images. Its overlays are not validated lesion diagnoses.
- Quality checks are heuristic and have not been validated across cameras or clinical settings.
- Splits and reported metrics should not be interpreted as patient-level or clinical validation unless explicitly established by the relevant experiment.
- More diverse, correctly labeled external data and independent validation are needed before making stronger performance claims.

## Ethics and intended use

This project is for education and research demonstration. It is not intended for patient care. Any future clinical use would require appropriate expert review, prospective evaluation, regulatory assessment, and validation across intended populations and imaging devices.
