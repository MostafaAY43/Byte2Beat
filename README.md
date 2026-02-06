# HeartGuard: ECG-Based Cardiovascular Risk Prediction System

A multi-modal deep learning system that analyzes 12-lead ECG signals to predict cardiovascular risk, explain its predictions, and generate personalized clinical intervention recommendations.

Built for the **Byte2Beat Hack4Health Hackathon 2026**.

## Overview

Cardiovascular disease is the leading cause of death worldwide, yet most ECG analysis tools stop at diagnosis. HeartGuard goes further by building a complete pipeline from raw ECG signal to clinical action: predicting patient-level risk, explaining which signal features drive each prediction, recommending evidence-based interventions with quantified risk reduction, and producing population-level analytics for healthcare resource planning.

## Key Features

- **Risk Prediction**: Composite cardiovascular risk scoring from ECG diagnoses, patient age, and sex, classified into four tiers (Normal, Low, Moderate, High)
- **4 Model Architectures**: XGBoost baseline, 1D CNN, CNN-LSTM hybrid, and Multi-Modal fusion (raw ECG + tabular features)
- **165 Engineered Features**: HRV time-domain metrics (SDNN, RMSSD, pNN50), per-lead statistical features, and morphological features extracted from raw 12-lead ECGs
- **Explainability**: SHAP values for feature importance (XGBoost) and Grad-CAM attention heatmaps showing which ECG regions drive CNN predictions
- **Clinical Recommendations**: Rule-based intervention system mapping diagnoses to categorized actions (immediate, medication, lifestyle, monitoring, referral) with urgency levels and expected risk reduction from clinical trial data
- **Population Analytics**: Cohort-level risk stratification for healthcare resource planning
- **Patient Reports**: Individualized risk reports with diagnoses, physiological metrics, intervention plans, and projected risk reduction

## Dataset

**PTB-XL** (Physikalisch-Technische Bundesanstalt): A large publicly available electrocardiography dataset.

| Property | Value |
|----------|-------|
| Total ECGs | 21,799 |
| Unique Patients | 18,869 |
| Sampling Rate | 500 Hz |
| Duration | 10 seconds per recording |
| Leads | 12-lead standard configuration |
| Age Range | 0-95 years (median 62) |
| Sex Distribution | 52% male, 48% female |

**Diagnostic Superclasses**:
| Records | Code | Description |
|---------|------|-------------|
| 9,514 | NORM | Normal ECG |
| 5,469 | MI | Myocardial Infarction |
| 5,235 | STTC | ST/T Change |
| 4,898 | CD | Conduction Disturbance |
| 2,649 | HYP | Hypertrophy |

Data split uses pre-defined patient-stratified folds: 1-8 for training, fold 9 for validation, fold 10 for testing (human-validated labels).

## Model Performance

Evaluated on the held-out test set (fold 10, n=2,198):

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | AUC |
|-------|----------|------------|---------------|-----|
| XGBoost | 0.6715 | 0.5536 | 0.6648 | 0.8030 |
| **1D CNN** | 0.6984 | **0.5765** | 0.6955 | **0.8345** |
| CNN-LSTM | 0.6683 | 0.5354 | 0.6807 | 0.8266 |
| **Multi-Modal** | **0.7029** | 0.5702 | **0.7027** | 0.8278 |

The 1D CNN achieves the best AUC (0.8345), while Multi-Modal fusion achieves the best accuracy (70.3%).

## Project Structure

```
Byte2Beat/
├── heartguard.ipynb              # Main notebook (end-to-end pipeline)
├── outputs/
│   ├── model_metrics.json        # Model performance metrics
│   ├── extracted_features.csv    # 165 features for all 21,799 ECGs
│   ├── test_predictions.csv      # Test set predictions from all models
│   ├── demographics.png          # Dataset demographics visualization
│   ├── diagnosis_distribution.png
│   ├── confusion_matrices.png    # All 4 models compared
│   ├── roc_curves.png            # ROC curves per class
│   ├── shap_summary.png          # SHAP feature importance
│   ├── gradcam_class_*.png       # Grad-CAM attention heatmaps
│   ├── population_analysis.png   # Population risk stratification
│   └── patient_report_*.md       # Sample patient reports
├── models/
|   ├── xgboost_model.pkl         # Trained XGBoost model
│   ├── cnn_model.pth             # Trained 1D CNN (PyTorch)
│   ├── cnn_lstm_model.pth        # Trained CNN-LSTM (PyTorch)
|   └── multimodal_model.pth      # Trained Multi-Modal (PyTorch)
└── HeartGuard.pdf                # Generated research paper includes findings and analysis
```

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3060 Ti, 4GB+ VRAM)

### Installation

```bash
pip install numpy pandas matplotlib seaborn plotly kaleido
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn xgboost shap wfdb scipy
```

### Dataset

Download PTB-XL from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and place it at `data/ptb-xl-1.0.3/`.

### Running

Open and run `heartguard.ipynb` end-to-end. The notebook handles data loading, feature extraction, model training, evaluation, explainability, and report generation.

## System Outputs

### Patient Risk Reports
Individualized reports containing risk category, detected diagnoses, HRV metrics, Grad-CAM attention analysis, and prioritized intervention plans with projected risk reduction.

### Intervention Recommendations
Evidence-based recommendations across five domains with quantified risk reduction:
- Statin therapy: 25-30% MACE reduction
- Smoking cessation: 50% CVD risk reduction
- Blood pressure control: 20-25% reduction per 10 mmHg
- Exercise program: 10-15% reduction

### Population Analytics
Cohort-level risk stratification by demographics, identification of highest-risk individuals, and intervention coverage analysis for resource planning.

## Limitations

- Risk labels are derived from ECG diagnoses, not true clinical outcomes (mortality, MACE)
- Clinical features (blood pressure, cholesterol) are synthetically generated
- Single-center dataset; generalization to diverse populations is unvalidated
- Requires prospective clinical validation before deployment

## Author

**Mostafa Abayazead** | Byte2Beat Hack4Health Hackathon 2026

## References

1. Wagner, P., et al. (2020). "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data*, 7(1), 154.
2. Lundberg, S. M. & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *NeurIPS*, 30.
3. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual explanations from deep networks." *ICCV*, 618-626.
