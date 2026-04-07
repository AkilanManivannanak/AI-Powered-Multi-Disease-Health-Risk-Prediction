<div align="center">

![AI_Medicalproject_coverimage](./AI_Medicalproject_coverimage.png)

![AI-Powered Multi-Disease Health Risk Prediction Banner](https://capsule-render.vercel.app/api?type=waving&color=0:202124,50:5f6368,100:dadce0&height=220&section=header&text=AI-Powered%20Multi-Disease%20Health%20Risk%20Prediction🏥&fontSize=38&fontColor=ffffff&fontAlignY=38&animation=fadeIn)

**Live repo:** [github.com/AKilalours/AI-Powered-Health-Risk-Prediction-System-for-Multi-Disease-Diagnosis](https://github.com/AKilalours/AI-Powered-Health-Risk-Prediction-System-for-Multi-Disease-Diagnosis)

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Diseases](https://img.shields.io/badge/diseases-17-blueviolet?style=flat-square)]()
[![Avg Accuracy](https://img.shields.io/badge/avg%20accuracy-87.3%25-brightgreen?style=flat-square)]()
[![Best Accuracy](https://img.shields.io/badge/best-100%25%20HIV-brightgreen?style=flat-square)]()

> ⚠️ **Academic project (LIU, AI 681). NOT a medical device. Not for clinical diagnosis. For research and education only.**

</div>

---

## 🎯 Goal & SLOs

Screen patients across **17 chronic and genetic conditions** from a single pipeline — combining tabular clinical data with chest X-ray imaging — and generate an interpretable, clinician-style risk report.

| SLO | Target | Achieved |
|---|---|---|
| **Disease coverage** | 17 conditions | **17** ✅ |
| **Average test accuracy** | > 85% | **87.3%** ✅ |
| **Best single-disease accuracy** | — | **98.94% (HIV)** ✅ |
| **TB inference latency (CPU)** | < 2s/image | **~1.2s/image** ✅ |
| **Full pipeline runtime** | — | **~2h 15m (Colab CPU)** |
| **Pneumonia CNN accuracy** | > 80% | ❌ 62.5% (active improvement — see postmortem) |

---

## 📐 Architecture

```
Patient Input
  ├── Tabular: labs, vitals, demographics (CSV / structured)
  └── Imaging: Chest X-ray (.jpg / .png)
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│                    PREPROCESSING                         │
│  Tabular:                                                │
│    • Missing value imputation                            │
│    • Numeric scaling (StandardScaler)                    │
│    • Categorical one-hot encoding                        │
│  Imaging:                                                │
│    • Resize + normalize                                  │
│    • ImageDataGenerator augmentation                     │
└──────────────────────┬───────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│  TABULAR PIPELINE    │   │  IMAGING PIPELINE            │
│  (15 diseases)       │   │  (2 diseases)                │
│                      │   │                              │
│  RandomForest        │   │  Pneumonia:                  │
│  n_estimators=200    │   │    EfficientNetB0 backbone   │
│  random_state=42     │   │    + GAP + Dropout + Dense   │
│  80/20 stratified    │   │                              │
│  train-test split    │   │  Tuberculosis:               │
│                      │   │    Custom 3-layer CNN        │
│  SHAP explainability │   │    Conv2D + MaxPool + Dense  │
└──────────┬───────────┘   └──────────────┬───────────────┘
           └─────────────┬────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              CLINICAL DECISION-SUPPORT REPORT            │
│                                                          │
│  • Per-disease probabilistic risk stratification         │
│  • Predicted progression (3–5 year horizon)              │
│  • Symptom timeline                                      │
│  • Disease interaction / comorbidity chain               │
│  • Immunity & resistance score                           │
│  • SHAP feature attribution                              │
│  • Radar chart (multi-disease risk map)                  │
│  • Clinical recommendations summary                      │
└──────────────────────────────────────────────────────────┘
```

**Trade-offs considered:**
- **Random Forest vs deep learning for tabular:** RF is fast, interpretable, and outperforms shallow NNs on structured clinical data with limited samples — chosen deliberately over gradient boosting for SHAP compatibility.
- **EfficientNetB0 for Pneumonia:** transfer learning reduces training data requirements; however, 62.5% accuracy signals domain shift between pretraining and the chest X-ray distribution — needs fine-tuning or dataset expansion.
- **Custom CNN for TB:** lightweight 3-layer CNN avoids overfitting on smaller TB dataset; trades max accuracy for generalisability. 96.9% achieved.
- **Latency vs scale:** full pipeline runs ~2h 15m on CPU — not production-ready. Modularisation + GPU would reduce to ~10–15 min. FastAPI serving is on the roadmap.

---

## Why This Matters

Most screening tools address a single disease. This system takes a **portfolio approach** — one pipeline, one patient record, 17 simultaneous risk assessments — mimicking how a clinician thinks about comorbidities rather than diseases in isolation.

Key design goals:
- **Interpretability first:** SHAP values make every prediction auditable, not a black box.
- **Clinician-readable output:** the report mirrors a clinical note structure — diagnosis → progression → recommendations.
- **Modular architecture:** each disease is an independent module; new conditions can be added without retraining the full system.

---

## Results (Test Accuracy — all 17 diseases)

**Average accuracy: 87.3%** across stratified 80/20 splits.

| Disease | Model Type | Test Accuracy |
|---|---|---:|
| HIV | Random Forest | **98.94%** |
| Heart Disease | Random Forest | **98.54%** |
| Chronic Kidney Disease | Random Forest | **97.56%** |
| Tuberculosis (X-ray) | Custom CNN | **96.90%** |
| Cancer | Random Forest | **96.49%** |
| Anemia | Random Forest | **95.46%** |
| Obesity | Random Forest | **95.27%** |
| Alzheimer's | Random Forest | **94.42%** |
| Stroke | Random Forest | **93.84%** |
| Parkinson's | Random Forest | **94.87%** |
| Hypertension | Random Forest | **88.44%** |
| Cystic Fibrosis | Random Forest | **84.21%** |
| Liver Disease | Random Forest | 76.07% |
| PCOS | Random Forest | 73.39% |
| Diabetes | Random Forest | 73.38% |
| Hepatitis | Random Forest | 70.97% |
| Pneumonia (X-ray) | EfficientNetB0 | ⚠️ 62.50% |

> Full accuracy table: Table 2 in `docs/fpr-Team-6.pdf`. Imaging model discussion in Section 4.

---

## 🔥 Postmortem: What Broke and How We Fixed It

### Issue 1 — Pneumonia CNN underperformed (62.5% accuracy)
**What happened:** EfficientNetB0 achieved only 62.5% on the pneumonia chest X-ray test set — well below the 80%+ target.

**Root cause:** Domain mismatch between ImageNet pretraining weights and chest X-ray features. The model was fine-tuned on too few epochs with limited augmentation, causing the backbone to retain ImageNet-biased filters rather than adapting to radiographic patterns.

**Fix applied:** Identified via per-class confusion analysis — the model was biased toward one class. Applied class-weighted loss and increased augmentation (shear, zoom, horizontal flip).

**Status:** Active improvement. Next step: unfreeze top layers of EfficientNetB0 for full fine-tuning + expand dataset with additional curated chest X-ray samples. Target: > 85%.

---

### Issue 2 — Class imbalance across multiple diseases
**What happened:** Several diseases (Hepatitis 70.97%, Diabetes 73.38%, PCOS 73.39%) showed lower accuracy consistent with class imbalance in the source datasets.

**Root cause:** Public datasets for rare or under-screened conditions tend to have imbalanced positive/negative ratios. Stratified splits help preserve proportions, but the model still sees fewer positive examples during training.

**Fix applied:** Stratified train-test splitting applied consistently across all 17 pipelines to prevent split-induced imbalance. Noted as a limitation in the report.

**Next steps:** Add SMOTE oversampling for underrepresented classes, evaluate with AUC-ROC and PR-AUC in addition to accuracy, and add calibration curves for probabilistic outputs.

---

### Issue 3 — Pipeline runtime (~2h 15m) not production-viable
**What happened:** Full pipeline across 17 models takes ~2h 15m on Colab CPU, making it impractical for interactive inference.

**Root cause:** RF training with n_estimators=200 across 15 models + CNN training (20 epochs pneumonia) is compute-heavy. No parallelisation was implemented; models trained sequentially.

**Fix applied:** Modularised each disease as an independent pipeline so individual models can be retrained independently. Pre-trained model serialisation (`.pkl` + `.h5`) enables inference-only mode at ~1–5s per patient.

**Next steps:** FastAPI `/predict/tabular` and `/predict/xray` endpoints + Docker container. GPU training reduces CNN time from ~1h 48m to ~5–10 min.

---

## ⚡ Runtime & Efficiency

Benchmarked on **Colab CPU (baseline)**:

| Stage | Time |
|---|---|
| Full pipeline (all 17 models, training) | ~2h 15m |
| Per RF model (avg) | ~4–6 min |
| Pneumonia CNN training (20 epochs) | ~1h 48m |
| TB inference (per image) | **~1.2s** |
| RF inference (per patient, pre-trained) | < 1s |

---

## Explainability & Visualisation

- **SHAP feature attribution** — per-prediction feature importance (e.g., glucose + BMI are top contributors for diabetes risk)
- **Radar chart** — multi-disease risk mapped across all 17 conditions for a single patient
- **Risk dashboards** — current diagnosed vs predicted progression vs future risk trajectory
- Screenshots of all output artefacts: Figures 19–27 in `docs/fpr-Team-6.pdf`

---

## 🛠️ MLOps & Infrastructure

### Reproducibility
- Fixed `random_state=42` across all RF classifiers.
- Stratified 80/20 splits enforced per disease — no cross-contamination between disease pipelines.
- All dependencies pinned in `requirements.txt`.

### Experiment tracking
- Results captured manually per run; accuracy table maintained in the report.
- Next step: **MLflow** or **Weights & Biases** for per-run metric logging, model versioning, and comparison across feature engineering variants.

### CI/CD awareness
- Planned: GitHub Actions workflow to run evaluation suite on PR merge — blocks merge if any disease accuracy regresses > 2% from baseline.
- Eval gates: promote model only if avg accuracy ≥ 87% AND no single disease drops below 60%.

### Deployment path
```
Train (Colab / local GPU)
        │
        ▼
Serialise → .pkl (RF) + .h5 (CNN)
        │
        ▼
FastAPI service
  ├── POST /predict/tabular  →  RF inference (<1s)
  └── POST /predict/xray     →  CNN inference (~1.2s)
        │
        ▼
Docker container → cloud deploy (GCP / AWS / Azure)
        │
        ▼
Observability: log per-request latency + prediction distribution
Alert if: avg confidence < 0.6 (data drift signal)
```

### Monitoring & alerts (planned)
- Log prediction confidence distribution per disease; alert on drift from training baseline.
- Shadow testing: new model variant runs alongside current, compared on a held-out validation set before promotion.
- Rollback: serialised model versions stored; rollback = swap `.pkl`/`.h5` path in config.

---

## 🔁 Reliability: Caching, Fallbacks, Observability

| Concern | Approach |
|---|---|
| **Slow CNN inference** | Pre-load model weights at startup; cache inference results for duplicate X-ray hashes |
| **Model load failure** | Fallback: return `risk: unknown, confidence: 0.0` with error flag rather than crashing |
| **Data drift** | Monthly evaluation on held-out real-world samples; alert if accuracy drops > 3% |
| **Eval gates** | Promote only if: avg acc ≥ 87% AND pneumonia acc > 80% AND no disease drops > 5% |
| **Rollback** | Versioned `.pkl` + `.h5` checkpoints; rollback = config path swap |

---

## 📦 Datasets (17 public sources)

All datasets are de-identified and publicly available. This repo does **not** redistribute raw data.

| # | Disease | Source |
|---|---|---|
| 1 | Diabetes | Pima Indians Diabetes Dataset |
| 2 | Liver Disease | Indian Liver Patient Records (Kaggle) |
| 3 | Parkinson's | UCI Parkinson's Dataset |
| 4 | PCOS | Kaggle PCOS Dataset |
| 5 | Stroke | Kaggle Stroke Prediction Dataset |
| 6 | Cystic Fibrosis | Kaggle CF Gene Expression |
| 7 | Alzheimer's | Kaggle Alzheimer's Dataset |
| 8 | Anemia | Kaggle Anemia Dataset |
| 9 | Chronic Kidney Disease | UCI CKD Dataset |
| 10 | Heart Disease | Kaggle Heart Disease (UCI) |
| 11 | Cancer | Wisconsin Breast Cancer Dataset (Kaggle) |
| 12 | Obesity | Kaggle Obesity Levels Dataset |
| 13 | Hypertension | Kaggle Hypertension Risk Data |
| 14 | Pneumonia | Chest X-ray Pneumonia (Kaggle) |
| 15 | Tuberculosis | TB Chest X-ray Dataset (Kaggle) |
| 16 | Hepatitis | UCI Hepatitis Dataset |
| 17 | HIV | Kaggle HIV/AIDS Dataset |

> See Appendix C in `docs/fpr-Team-6.pdf` for full dataset citations and licensing notes.

---

## 🚀 Quickstart

```bash
git clone https://github.com/AKilalours/AI-Powered-Health-Risk-Prediction-System-for-Multi-Disease-Diagnosis
cd AI-Powered-Health-Risk-Prediction-System-for-Multi-Disease-Diagnosis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline (adjust patient args as needed)
python src/run_all.py --patient_id KP20250503-001 --age 48 --gender Male
```

**Run on Colab:**
- Open the notebook and mount Google Drive if datasets are stored there.
- Execute end-to-end to reproduce metrics and report screenshots.

---

## 📁 Repository Structure

```
AI-Powered-Health-Risk-Prediction-System-for-Multi-Disease-Diagnosis/
├── src/
│   ├── run_all.py                  # Main entrypoint
│   ├── tabular/                    # Per-disease RF pipelines (15 modules)
│   └── imaging/                    # CNN pipelines (Pneumonia + TB)
├── docs/
│   └── fpr-Team-6.pdf              # Full report (methodology + results)
├── data/
│   └── README.md                   # Dataset provenance + download instructions
├── models/                         # Serialised .pkl + .h5 checkpoints
├── requirements.txt
└── README.md
```

---

## Responsible Use & Limitations

- This is a **screening and decision-support prototype** trained on multiple independent public datasets — not a unified hospital EHR distribution.
- **Do not use for diagnosis or treatment decisions.**
- Known limitations: dataset heterogeneity, class imbalance in several diseases, no real-time API (yet), Pneumonia accuracy below target.
- Full limitations discussion: Section 5 of `docs/fpr-Team-6.pdf`.

---

## 🗺️ Roadmap

- Add cross-validation + AUC/PR-AUC + calibration curves (priority: imbalanced diseases).
- Add leakage checks, strict split hygiene, and dataset versioning.
- Build FastAPI service (`/predict/tabular`, `/predict/xray`) + Dockerfile.
- Track experiments with MLflow / W&B; create eval regression suite.
- Add unit tests for preprocessing + report generation and CI via GitHub Actions.
- Fine-tune EfficientNetB0 for Pneumonia to reach > 85%.

---

## 📄 Report

Full methodology, ablation analysis, dataset citations, and result screenshots: [`docs/fpr-Team-6.pdf`](./docs/fpr-Team-6.pdf)

---

## 👥 Team & Contributions

This is a collaborative team project — all work was split equally between both members.

| | **Akila Lourdes Miriyala Francis** | **Akilan Manivannan** |
|---|---|---|
| 📦 **Data & Preprocessing** | Dataset sourcing for diseases 1–9 (Diabetes → CKD), imputation & scaling pipeline | Dataset sourcing for diseases 10–17 (Heart → HIV), categorical encoding & augmentation |
| 🧠 **Modelling — Tabular** | RF pipelines: Diabetes, Liver, Parkinson's, PCOS, Stroke, Cystic Fibrosis, Alzheimer's, Anemia | RF pipelines: CKD, Heart Disease, Cancer, Obesity, Hypertension, Hepatitis, HIV |
| 🏗️ **Modelling — Imaging** | EfficientNetB0 Pneumonia CNN pipeline, ImageDataGenerator augmentation | Custom 3-layer TB CNN architecture, Conv2D + MaxPool + Dense design |
| 📊 **Evaluation** | Classification reports + accuracy tables for tabular diseases, confusion matrix analysis | SHAP explainability integration, radar chart generation, per-image CNN evaluation |
| ⚡ **Efficiency & Ops** | Runtime benchmarking (per-model timing), model serialisation (`.pkl` / `.h5`) | Pipeline orchestration (`run_all.py`), inference-only mode, Colab integration |
| 📝 **Documentation** | Architecture diagrams, postmortem write-up, roadmap | Report compilation (`fpr-Team-6.pdf`), dataset appendix, results tables |

> **Equal contribution** — all design decisions, model selection, and evaluation criteria were discussed and validated jointly.

---

<div align="center">
<sub>Built with scikit-learn · TensorFlow · SHAP · EfficientNetB0 · Random Forest</sub><br/>
<sub>Academic project (LIU, AI 681) · Akila Lourdes Miriyala Francis & Akilan Manivannan</sub>
</div>
