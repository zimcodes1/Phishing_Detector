# Model Training and Evaluation Documentation

## 1. Purpose of This Stage

This document records the model training and model evaluation phases of the phishing detection project. It follows the completed feature importance extraction stage documented in `docs/feature_importance_extraction.md`.

The three project goals are:

1. Feature importance extraction: select the 5-7 most important features.
2. Model training: train phishing detection models using the selected features.
3. Model evaluation: evaluate trained models using formal classification metrics.

This document covers goals 2 and 3. It explains how the models were trained, which selected features were used, what the evaluation metrics mean, how each model performed, and which output files were generated in the `outputs/`, `models/`, and `datasets/` directories.

## 2. Notebooks Reviewed

The model training and evaluation stages were implemented in four notebooks:

| Notebook | Purpose |
| --- | --- |
| `notebooks/url_xgb-rf-model-training.ipynb` | Trains URL phishing models using the selected top-7 URL features. |
| `notebooks/email-xgb-rf-model-training.ipynb` | Trains email phishing models using the selected top-7 email features. |
| `notebooks/url-xgb-rf-model-evaluation.ipynb` | Evaluates trained URL models on the saved held-out URL test set. |
| `notebooks/email-phishing-xgb-rf-model-evaluation.ipynb` | Evaluates trained email models on the saved held-out email test set. |

The workflow intentionally separates training from evaluation. The training notebooks tune and save the models, while the evaluation notebooks reload the saved models and the exact saved test sets. This avoids re-splitting the data during evaluation and makes the reported test results reproducible.

## 3. Selected Features Used for Training

Only the top-7 features from the feature importance stage were used for model training.

### 3.1 URL Selected Features

The URL model used:

| Rank | Feature |
| ---: | --- |
| 1 | `HTTPS` |
| 2 | `AnchorURL` |
| 3 | `PrefixSuffix-` |
| 4 | `ServerFormHandler` |
| 5 | `WebsiteTraffic` |
| 6 | `GoogleIndex` |
| 7 | `DNSRecording` |

The URL target label was remapped from the original dataset convention:

| Original Label | Remapped Label | Meaning |
| ---: | ---: | --- |
| `-1` | `0` | Phishing |
| `1` | `1` | Legitimate |

Therefore, for the URL model, phishing is class `0`.

### 3.2 Email Selected Features

The email model used:

| Rank | Feature |
| ---: | --- |
| 1 | `body_length` |
| 2 | `body_html_tag_count` |
| 3 | `body_caps_ratio` |
| 4 | `urls_present` |
| 5 | `body_urgency_count` |
| 6 | `body_link_count` |
| 7 | `body_exclamation_count` |

The email dataset already used a binary label convention:

| Label | Meaning |
| ---: | --- |
| `0` | Legitimate |
| `1` | Phishing |

Therefore, for the email model, phishing is class `1`.

## 4. Train/Test Split

Both datasets used a stratified 80/20 train/test split with `random_state=42`. Stratification preserved the original class balance in the train and test sets.

### 4.1 URL Dataset Split

| Split | Shape | Class 0: Phishing | Class 1: Legitimate |
| --- | ---: | ---: | ---: |
| Training | 8,843 rows x 7 features | 3,918 (44.31%) | 4,925 (55.69%) |
| Testing | 2,211 rows x 7 features | 979 (44.28%) | 1,232 (55.72%) |

### 4.2 Email Dataset Split

| Split | Shape | Class 0: Legitimate | Class 1: Phishing |
| --- | ---: | ---: | ---: |
| Training | 31,323 rows x 7 features | 13,850 (44.22%) | 17,473 (55.78%) |
| Testing | 7,831 rows x 7 features | 3,462 (44.21%) | 4,369 (55.79%) |

## 5. Models Trained

Two supervised machine learning models were trained for each dataset:

1. XGBoost Classifier.
2. Random Forest Classifier.

These models were selected because both are strong tree-based classifiers, can model non-linear relationships, and provide competitive performance on tabular classification tasks.

### 5.1 XGBoost Classifier

XGBoost is a gradient boosting algorithm that builds decision trees sequentially. Each new tree attempts to correct errors made by previous trees. It is often effective for structured/tabular datasets because it can capture interactions between features.

In this project, XGBoost was tuned using `GridSearchCV` over:

| Hyperparameter | Candidate Values |
| --- | --- |
| `n_estimators` | `100`, `300`, `500` |
| `max_depth` | `3`, `6`, `9` |
| `learning_rate` | `0.01`, `0.1`, `0.2` |

The model used `eval_metric="logloss"` and `random_state=42`. The training notebooks also computed `scale_pos_weight` from the class ratio to account for mild class imbalance.

### 5.2 Random Forest Classifier

Random Forest is an ensemble learning method that trains many decision trees and combines their predictions. It reduces the instability of single decision trees by averaging many trees trained on different samples and feature splits.

Random Forest was tuned using `GridSearchCV` over:

| Hyperparameter | Candidate Values |
| --- | --- |
| `n_estimators` | `100`, `300`, `500` |
| `max_depth` | `None`, `10`, `20` |
| `min_samples_split` | `2`, `5`, `10` |

The model used `class_weight="balanced"` and `random_state=42`.

## 6. Hyperparameter Tuning Method

Both models were tuned using 5-fold `GridSearchCV`. The scoring metric during tuning was F1-score for the phishing class:

- URL dataset: F1 with `pos_label=0`, because phishing is class `0`.
- Email dataset: F1 with `pos_label=1`, because phishing is class `1`.

F1-score was chosen for tuning because phishing detection requires a balance between:

- catching phishing cases, measured by recall;
- avoiding false alarms, measured by precision.

Accuracy alone was not used for tuning because it can be misleading when one class is more common than the other.

## 7. Best Hyperparameters and Cross-Validation Scores

### 7.1 URL Model Training Results

| Model | Best Hyperparameters | Best CV F1 for Phishing | Fold Scores | Mean ± Std |
| --- | --- | ---: | --- | --- |
| XGBoost | `learning_rate=0.1`, `max_depth=3`, `n_estimators=300` | 0.9175 | 0.9173, 0.9126, 0.9263, 0.9136, 0.9177 | 0.9175 ± 0.0048 |
| Random Forest | `max_depth=None`, `min_samples_split=5`, `n_estimators=100` | 0.9185 | 0.9232, 0.9120, 0.9248, 0.9136, 0.9191 | 0.9185 ± 0.0051 |

The URL Random Forest model had a slightly higher cross-validation F1-score than XGBoost, but the difference was very small. Both models were stable across folds, with standard deviations around 0.005.

Training-set sanity check:

| Model | Training Accuracy | Training F1 for Phishing |
| --- | ---: | ---: |
| XGBoost | 92.84% | 91.89% |
| Random Forest | 92.93% | 92.07% |

These training scores were used only to confirm that the models were learning. The formal performance judgment is based on the held-out test set.

### 7.2 Email Model Training Results

| Model | Best Hyperparameters | Best CV F1 for Phishing | Fold Scores | Mean ± Std |
| --- | --- | ---: | --- | --- |
| XGBoost | `learning_rate=0.1`, `max_depth=9`, `n_estimators=300` | 0.9699 | 0.9667, 0.9713, 0.9715, 0.9699, 0.9701 | 0.9699 ± 0.0017 |
| Random Forest | `max_depth=None`, `min_samples_split=2`, `n_estimators=100` | 0.9718 | 0.9690, 0.9742, 0.9742, 0.9696, 0.9717 | 0.9718 ± 0.0022 |

The email Random Forest model also achieved the higher cross-validation F1-score, again by a small margin. Both models showed strong and stable cross-validation performance.

Training-set sanity check:

| Model | Training Accuracy | Training F1 for Phishing |
| --- | ---: | ---: |
| XGBoost | 98.69% | 98.83% |
| Random Forest | 99.98% | 99.98% |

The Random Forest training score is extremely high and may indicate a very strong fit to the training data. The held-out test metrics are therefore more important for judging whether the model generalizes.

## 8. Evaluation Metrics Explained

The evaluation notebooks computed accuracy, precision, recall, F1-score, ROC-AUC, classification reports, confusion matrices, ROC curves, and precision-recall curves.

### 8.1 Confusion Matrix Terms

For phishing detection, the important class is phishing. The confusion matrix can be interpreted as:

| Term | Meaning in This Project |
| --- | --- |
| True Positive (TP) | A phishing sample correctly predicted as phishing. |
| False Positive (FP) | A legitimate sample incorrectly predicted as phishing. |
| True Negative (TN) | A legitimate sample correctly predicted as legitimate. |
| False Negative (FN) | A phishing sample incorrectly predicted as legitimate. |

False negatives are especially important because they represent phishing cases that the model failed to detect.

### 8.2 Accuracy

Accuracy is the proportion of all predictions that were correct:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Accuracy is easy to understand, but it can hide poor phishing detection if the dataset is imbalanced.

### 8.3 Precision

Precision measures how many samples predicted as phishing were actually phishing:

```text
Precision = TP / (TP + FP)
```

High precision means the model produces fewer false alarms.

### 8.4 Recall

Recall measures how many actual phishing samples the model successfully found:

```text
Recall = TP / (TP + FN)
```

High recall is important in phishing detection because missed phishing attempts can be harmful.

### 8.5 F1-Score

F1-score is the harmonic mean of precision and recall:

```text
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

F1 is useful when both false positives and false negatives matter. It was the main model selection metric in this project.

### 8.6 ROC-AUC

ROC-AUC measures how well the model ranks positive samples above negative samples across classification thresholds. A value near `1.0` indicates strong separability, `0.5` indicates random ranking, and values below `0.5` usually indicate that the score direction or positive-class convention may be inverted.

### 8.7 Precision-Recall Curve and Average Precision

The precision-recall curve shows the trade-off between precision and recall across thresholds. It is especially useful when the positive class is important, as it is here. Average Precision summarizes the precision-recall curve into one score.

## 9. URL Model Evaluation Results

The URL evaluation used the saved held-out test set of 2,211 samples.

### 9.1 URL Test Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9344 | 0.9281 | 0.9234 | 0.9258 | 0.9831 |
| Random Forest | 0.9344 | 0.9204 | 0.9326 | 0.9264 | 0.9836 |

The two URL models achieved the same overall accuracy of `93.44%`. Random Forest had the slightly higher phishing recall, F1-score, and ROC-AUC, while XGBoost had the slightly higher precision.

Approximate confusion-matrix interpretation from the saved evaluation results:

| Model | Phishing Correctly Detected | Phishing Missed | Legitimate Correctly Detected | Legitimate Flagged as Phishing |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | ~904 | ~75 | ~1,162 | ~70 |
| Random Forest | ~913 | ~66 | ~1,153 | ~79 |

This means Random Forest caught slightly more phishing cases, but it also produced slightly more false phishing alarms than XGBoost.

### 9.2 URL ROC-AUC Interpretation

The updated URL ROC-AUC values are `0.9831` for XGBoost and `0.9836` for Random Forest. These values are consistent with the high accuracy, precision, recall, and F1 scores reported for both URL models.

Because URL phishing is encoded as class `0`, the ROC-AUC calculation must use the probability score aligned with the phishing class rather than blindly assuming class `1` is always the positive class. The corrected scores indicate that both URL models separate phishing and legitimate URLs very well across classification thresholds.

### 9.3 URL Evaluation Finding

The URL results show that both XGBoost and Random Forest performed well using only the seven selected URL features. Random Forest is the slightly stronger model if phishing recall and F1-score are prioritized. XGBoost is slightly better if minimizing false positives is prioritized.

## 10. Email Model Evaluation Results

The email evaluation used the saved held-out test set of 7,831 samples.

### 10.1 Email Test Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.9645 | 0.9623 | 0.9746 | 0.9684 | 0.9929 |
| Random Forest | 0.9663 | 0.9679 | 0.9718 | 0.9698 | 0.9924 |

Both email models performed very strongly. Random Forest achieved the best accuracy, precision, and F1-score. XGBoost achieved the best recall and ROC-AUC.

Approximate confusion-matrix interpretation from the saved evaluation results:

| Model | Phishing Correctly Detected | Phishing Missed | Legitimate Correctly Detected | Legitimate Flagged as Phishing |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | ~4,258 | ~111 | ~3,295 | ~167 |
| Random Forest | ~4,246 | ~123 | ~3,321 | ~141 |

XGBoost detected slightly more phishing emails, while Random Forest produced fewer false positives and achieved the stronger overall F1-score.

### 10.2 Email Evaluation Finding

The email models generalized strongly to the held-out test set. The high ROC-AUC values, both above `0.99`, indicate strong separation between phishing and legitimate emails. Random Forest is the best overall email model by F1-score, while XGBoost is preferable if the priority is maximum phishing recall.

## 11. Comparative Model Findings

### 11.1 Best Model by Dataset

| Dataset | Best Model by F1 | F1 | Reason |
| --- | --- | ---: | --- |
| URL | Random Forest | 0.9264 | Slightly higher phishing recall and F1 than XGBoost. |
| Email | Random Forest | 0.9698 | Best accuracy, precision, and F1 among email models. |

### 11.2 Main Observations

1. Both datasets achieved strong performance using only seven selected features.
2. Random Forest slightly outperformed XGBoost by F1-score on both URL and email test sets.
3. XGBoost remained competitive and, for the email dataset, achieved the highest phishing recall.
4. The email models performed better than the URL models overall.
5. The email selected features, especially body length, HTML tag count, URL presence, link count, uppercase ratio, urgency terms, and exclamation marks, provided strong predictive power.
6. URL phishing detection also performed well, but the lower F1-score compared with email suggests the selected URL feature set may leave out some useful URL indicators from the full 30-feature dataset.
7. The corrected URL ROC-AUC values are high for both models, with Random Forest slightly ahead of XGBoost.

## 12. Generated Outputs and Artifacts

### 12.1 Trained Model Files

| File | Description |
| --- | --- |
| `models/url/xgb_model.joblib` | Saved tuned XGBoost model for URL phishing detection. |
| `models/url/rf_model.joblib` | Saved tuned Random Forest model for URL phishing detection. |
| `models/email/xgb_model_email.joblib` | Saved tuned XGBoost model for email phishing detection. |
| `models/email/rf_model_email.joblib` | Saved tuned Random Forest model for email phishing detection. |

### 12.2 Saved Test Sets and Hyperparameters

| File | Description |
| --- | --- |
| `datasets/url/X_test.csv` | Held-out URL test features used during evaluation. |
| `datasets/url/y_test.csv` | Held-out URL test labels used during evaluation. |
| `datasets/url/best_params.json` | Best URL model hyperparameters from training. |
| `datasets/email/X_test_email.csv` | Held-out email test features used during evaluation. |
| `datasets/email/y_test_email.csv` | Held-out email test labels used during evaluation. |
| `datasets/email/best_params_email.json` | Best email model hyperparameters from training. |

### 12.3 URL Evaluation Outputs

| File | Description |
| --- | --- |
| `outputs/url_models_evaluation/url_test_metrics.csv` | CSV table containing URL test accuracy, precision, recall, F1, and ROC-AUC. |
| `outputs/url_models_evaluation/url_evaluation_summary.json` | JSON summary of URL evaluation results. |
| `outputs/url_models_evaluation/url_confusion_matrices.png` | Confusion-matrix plot for URL XGBoost and Random Forest models. |
| `outputs/url_models_evaluation/url_roc_curve.png` | ROC curve plot for URL models. |
| `outputs/url_models_evaluation/url_precision_recall_curve.png` | Precision-recall curve plot for URL models. |

### 12.4 Email Evaluation Outputs

| File | Description |
| --- | --- |
| `outputs/email_models_evaluation/email_test_metrics.csv` | CSV table containing email test accuracy, precision, recall, F1, and ROC-AUC. |
| `outputs/email_models_evaluation/email_evaluation_summary.json` | JSON summary of email evaluation results. |
| `outputs/email_models_evaluation/email_confusion_matrices.png` | Confusion-matrix plot for email XGBoost and Random Forest models. |
| `outputs/email_models_evaluation/email_roc_curve.png` | ROC curve plot for email models. |
| `outputs/email_models_evaluation/email_precision_recall_curve.png` | Precision-recall curve plot for email models. |

### 12.5 Logs

| File | Description |
| --- | --- |
| `outputs/logs/phishing-detection-xgb-rf-model-training.log` | URL model training log with selected features, split details, best parameters, CV scores, and training sanity checks. |
| `outputs/logs/url-phishing-xgb-rf-model-evaluation.log` | URL model evaluation log with predictions, metrics, classification reports, and saved artifact paths. |
| `outputs/logs/email-phishing-detection-xgb-rf-model-training.log` | Email model training log with selected features, split details, best parameters, CV scores, and training sanity checks. |
| `outputs/logs/email-phishing-xgb-rf-model-evaluation.log` | Email model evaluation log with predictions, metrics, classification reports, and saved artifact paths. |

## 13. Statement of Findings

The model training and evaluation phases produced the following findings:

1. Both XGBoost and Random Forest were successfully trained on the selected top-7 feature subsets for URL and email phishing detection.
2. The stratified split preserved class balance in both datasets, supporting fairer model training and evaluation.
3. Hyperparameter tuning was performed with 5-fold cross-validation using phishing-class F1-score as the optimization target.
4. Random Forest achieved the best test F1-score on both datasets.
5. For URL phishing detection, Random Forest achieved `accuracy=0.9344`, `precision=0.9204`, `recall=0.9326`, and `F1=0.9264`.
6. For URL phishing detection, XGBoost achieved the same accuracy as Random Forest but slightly lower F1-score.
7. For email phishing detection, Random Forest achieved `accuracy=0.9663`, `precision=0.9679`, `recall=0.9718`, and `F1=0.9698`.
8. For email phishing detection, XGBoost achieved the best recall at `0.9746`, meaning it detected the largest proportion of phishing emails.
9. The email models outperformed the URL models overall, suggesting the selected email body/link/style features were highly informative for this dataset.
10. The corrected URL ROC-AUC results support the same conclusion as the F1-score: Random Forest is slightly stronger overall for URL phishing detection.
11. The saved outputs are sufficient for report compilation, including metrics tables, summary JSON files, confusion matrices, ROC curves, and precision-recall curves.

## 14. Conclusion

The model training and evaluation phases successfully completed the second and third project goals. The selected top-7 features were sufficient to train strong phishing detection models for both URL and email datasets. Random Forest was the strongest overall model by F1-score in both cases, while XGBoost remained competitive and offered strong recall, particularly for email phishing detection.

For the final PGD report, the recommended performance summary is:

| Dataset | Recommended Model | Main Justification |
| --- | --- | --- |
| URL phishing | Random Forest | Best phishing-class F1 and recall among URL models. |
| Email phishing | Random Forest | Best overall F1, accuracy, and precision among email models. |

If the project prioritizes minimizing missed phishing cases above all else, XGBoost may also be discussed for the email dataset because it produced the highest email phishing recall.
