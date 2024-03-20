# potassium validation


## I. Input preparation


1. Prerequisites

    Ensure all ECG files are stored as *.npy in a single flat directory.
    Accompany these files with a manifest CSV file in the same directory.


2. Manifest File Format

    The manifest CSV file should include a header.
    Each row corresponds to one ECG file with the following columns:

- filename: Name of the .npy file (without the extension).
- age: Numeric value.
- HTN: Hypertension indicator.
- DM: Diabetes Mellitus indicator.
- dyslipidemia: Dyslipidemia indicator.
- smoke: Smoking status (1 if smoker).
- sex: Gender (1 if male).
- chest_pain_type: Type of chest pain (2 for Typical, 1 for Atypical, 0 for None).
- PCI: Percutaneous Coronary Intervention indicator.



## II. Inference


### A. ECG model inference


1. Use predict.py for the initial ECG model inference.

2. Set the following paths in predict.py:
data_path: Path to the directory containing ECG .npy files.
manifest_path: Path to the manifest CSV file.
weights: Path to the model weight file, which should be "ecg_model_best_epoch_val_mean_roc_auc_11_20_2023.pt".

3. Execute predict.py.

4. After execution, a file named "dataloader_0_predictions.csv" will be generated in the same directory. This file includes your original manifest data with an additional column "preds" indicating the ECG model predictions.


### B. CAD model inference


1. For further inference using the CAD model, use cad_predict.py.

2. Set manifest_path in cad_predict.py to the path of the predictions.csv file generated from Step A. Set weight_path to path of "nn_for_cad_prediction_in5years_ecg_cad2_best_1_6_2024.pth".

3. Run the cad_predict.py script.

4. Upon completion, a file named "cad_predictions.csv" will be saved in the same directory. This file contains the inference results "cad_preds" from the CAD model.
