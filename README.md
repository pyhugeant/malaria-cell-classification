# Malaria Cell Classification (NIH Dataset)

This project implements a complete deep learning pipeline for binary classification of malaria-infected red blood cells using convolutional neural networks.

---

## 1. Task Description

The objective is to classify individual red blood cell (RBC) images into:

* **Parasitized (infected)**
* **Uninfected**

This task is clinically relevant for automated malaria screening and large-scale diagnostic support.

---

## 2. Dataset

* Source: NIH Malaria Cell Images Dataset
* Size: ~27,500 RGB images
* Format: Cropped single-cell images

Directory structure:

```
data/malaria/
  Parasitized/
  Uninfected/
```

The dataset is randomly split into:

* Training: 70%
* Validation: 15%
* Test: 15%

---

## 3. Model

Baseline model:

* **ResNet-50 (pretrained on ImageNet)**
* Modified final layer for binary classification

Loss:

* CrossEntropyLoss

---

## 4. Evaluation Metrics

We evaluate performance using **Parasitized as the positive class**:

* Precision
* Recall
* F1-score
* Specificity

Definitions:

* **Precision**: How many predicted infected cells are truly infected
* **Recall**: How many infected cells are detected
* **Specificity**: How well normal cells are correctly identified

---

## 5. Results

| Metric      | Value  |
| ----------- | ------ |
| Accuracy    | 0.9727 |
| Precision   | 0.9770 |
| Recall      | 0.9668 |
| F1 Score    | 0.9718 |
| Specificity | 0.9783 |

Confusion summary:

* TP: 1953
* TN: 2069
* FP: 46
* FN: 67

The model achieves a strong balance between sensitivity and specificity with low false positive and false negative rates.

---

## 6. Observations

* The model converges quickly and achieves optimal performance in early epochs
* Slight overfitting appears in later training stages
* Data loading from local runtime significantly improves training speed compared to Google Drive
* Predictions are highly confident, indicating strong feature separability

---

## 7. Training

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train model

```bash
python train.py --config configs/malaria_resnet50.yaml
```

---

## 8. Inference

Single image prediction:

```bash
python infer.py \
  --config configs/malaria_resnet50.yaml \
  --checkpoint checkpoints/best.pt \
  --image sample.png
```

Example output:

```
Predicted class: Parasitized
Confidence: 1.0000
Probability Parasitized: 1.0000
Probability Uninfected: 0.0000
```

---

## 9. Full Test Set Inference

To run inference on the entire test set and export predictions:

```bash
python scripts/run_test_inference.py
```

Output:

```
runs/malaria-cell-classification/test_predictions.csv
```

This file contains:

* true labels
* predicted labels
* class probabilities

---

## 10. Visualization

### Training Curve

![Training Curve](runs/malaria-cell-classification/training_curve.png)

### Confusion Matrix

![Confusion Matrix](runs/malaria-cell-classification/confusion_matrix.png)

---

## 11. Project Structure

```
configs/        # experiment configs
src/
  datasets/     # data loading
  models/       # model definitions
  eval/         # metrics
  utils/        # utilities
scripts/        # evaluation / inference scripts
train.py        # training entry
infer.py        # single image inference
```

---

## 12. Reproducibility

* Fixed random seed
* Config-based experiment control
* Deterministic dataset split

---

## 13. Future Work

* Compare lightweight architectures (ResNet18, MobileNet)
* Apply probability calibration (reduce overconfidence)
* Add ROC curve and AUC evaluation
* Extend to multi-class blood cell classification

---

## Author

This project demonstrates an end-to-end deep learning workflow for medical image classification, including training, evaluation, and deployment-ready inference.
