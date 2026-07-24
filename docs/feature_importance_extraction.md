# Feature Importance Extraction Documentation

## 1. Purpose of This Stage

This document records the feature importance extraction stage of the phishing detection research workflow. The wider project is organized around three major goals:

1. Feature importance extraction: identify the 5-7 most important features.
2. Model training: train phishing detection models using the selected feature set.
3. Model evaluation: evaluate the trained models using appropriate classification metrics.

This stage addresses the first goal. Its purpose is to reduce the original feature spaces into compact, interpretable feature subsets that can be carried forward into the model training phase. The process was implemented in two notebooks:

- `notebooks/feature_importance_url.ipynb` for URL-based phishing detection.
- `notebooks/feature_importance_email.ipynb` for email-based phishing detection.

The extraction process used XGBoost gain-based feature importance as the primary selection method and SHAP values as a secondary interpretability and ranking-consistency check. The final output of this stage is a top-7 feature subset for each dataset.

## 2. Rationale for Feature Importance Extraction

Feature importance extraction is a necessary preparatory step in this project because phishing detection datasets often contain multiple overlapping indicators. Some features may provide strong discriminative power, while others may add only marginal information or noise. Selecting the most important features supports the research in the following ways:

- It improves interpretability by identifying the signals most responsible for distinguishing phishing from legitimate samples.
- It reduces dimensionality before the next training phase.
- It provides a defensible basis for explaining why certain URL or email properties were retained.
- It supports later reporting by connecting model behavior to observable phishing characteristics.
- It can reduce unnecessary computational cost during model training and evaluation.

Because this is a PGD research project, the selected features should not be treated only as technical variables. They also provide empirical evidence about the phishing indicators that were most influential in the datasets studied.

## 3. Data Sources and Dataset Scope

### 3.1 URL Phishing Dataset

The URL notebook used the Kaggle dataset "Phishing website Detector" by `eswarchandt`. The notebook searches for `phishing.csv` either in the local working directory or inside `/kaggle/input`.

Observed dataset properties from the executed notebook output:

| Property | Value |
| --- | ---: |
| Raw shape | 11,054 rows x 32 columns |
| Shape after dropping `Index` | 11,054 rows x 31 columns |
| Number of model features | 30 |
| Target column | `class` |
| Original labels | `1` and `-1` |
| Remapped labels | `1` = legitimate, `0` = phishing |
| Missing values after preprocessing | 0 |

The original URL class distribution was moderately balanced:

| Original class | Count | Percentage |
| --- | ---: | ---: |
| `1` | 6,157 | 55.70% |
| `-1` | 4,897 | 44.30% |

After remapping, the same distribution became:

| Remapped class | Count | Percentage |
| --- | ---: | ---: |
| `1` | 6,157 | 55.70% |
| `0` | 4,897 | 44.30% |

### 3.2 Email Phishing Dataset

The email notebook used the CEAS email phishing dataset, loaded from `CEAS_08.csv`. The notebook searches for the file locally or under `/kaggle/input`.

Observed dataset properties from the executed notebook output:

| Property | Value |
| --- | ---: |
| Raw shape | 39,154 rows x 7 columns |
| Original columns | `sender`, `receiver`, `date`, `subject`, `body`, `label`, `urls` |
| Engineered feature matrix shape | 39,154 rows x 17 columns |
| Number of engineered model features | 16 |
| Target column | `label` |
| Missing values in engineered features | 0 |
| Infinite values in engineered features | 0 |

The original email label distribution was also moderately balanced:

| Label | Count | Percentage |
| --- | ---: | ---: |
| `1` | 21,842 | 55.78% |
| `0` | 17,312 | 44.22% |

The notebook retained the dataset's existing `0` and `1` label convention because the observed labels were already binary.

## 4. Preprocessing and Feature Preparation

### 4.1 URL Dataset Preparation

The URL dataset already contained structured numeric phishing indicators. The preprocessing steps were therefore minimal:

1. Locate and load `phishing.csv`.
2. Drop the `Index` column because it is a row identifier rather than a predictive feature.
3. Confirm data types and null counts.
4. Remap the target column from `-1`/`1` to `0`/`1` for XGBoost compatibility.
5. Separate the feature matrix `X` from the target vector `y`.
6. Perform a stratified 80/20 train/test split using `random_state=42`.

The URL split preserved the original class proportions:

| Split | Class 0 Count | Class 0 % | Class 1 Count | Class 1 % |
| --- | ---: | ---: | ---: | ---: |
| Train | 3,918 | 44.31% | 4,925 | 55.69% |
| Test | 979 | 44.28% | 1,232 | 55.72% |

The final URL training shape was `(8843, 30)`, and the test shape was `(2211, 30)`.

### 4.2 Email Dataset Preparation and Feature Engineering

Unlike the URL dataset, the email dataset contained raw textual and metadata fields. The notebook therefore engineered features from sender information, receiver information, date values, subject text, body text, and URL presence.

The engineered email features were:

| Feature | Description |
| --- | --- |
| `sender_is_freemail` | Indicates whether the sender domain belongs to a common free email provider such as Gmail, Yahoo, Outlook, Hotmail, or AOL. |
| `sender_domain_length` | Length of the extracted sender domain. |
| `domain_mismatch` | Indicates whether sender and receiver domains differ when both are available. |
| `email_hour` | Hour extracted from the email date field, with missing values filled using the median hour. |
| `is_weekend` | Indicates whether the parsed email date falls on a Saturday or Sunday. |
| `subject_length` | Character length of the subject. |
| `subject_urgency_count` | Count of selected urgency-related keywords in the subject. |
| `subject_exclamation_count` | Number of exclamation marks in the subject. |
| `subject_caps_ratio` | Ratio of uppercase alphabetic characters in the subject. |
| `body_length` | Character length of the email body. |
| `body_urgency_count` | Count of selected urgency-related keywords in the body. |
| `body_html_tag_count` | Count of HTML-like tags in the body. |
| `body_link_count` | Count of `http` or `www` occurrences in the body. |
| `body_exclamation_count` | Number of exclamation marks in the body. |
| `body_caps_ratio` | Ratio of uppercase alphabetic characters in the body. |
| `urls_present` | Numeric value from the dataset's `urls` column indicating URL presence/count. |

The urgency keyword list used in the notebook was:

```text
urgent, verify, suspended, action required, password, confirm, immediately, click here, account
```

After feature engineering, the notebook confirmed that the engineered matrix contained no missing or infinite values. A stratified 80/20 split was then applied using `random_state=42`.

The email split preserved the original class proportions:

| Split | Class 0 Count | Class 0 % | Class 1 Count | Class 1 % |
| --- | ---: | ---: | ---: | ---: |
| Train | 13,850 | 44.22% | 17,473 | 55.78% |
| Test | 3,462 | 44.21% | 4,369 | 55.79% |

The final email training shape was `(31323, 16)`, and the test shape was `(7831, 16)`.

## 5. Feature Importance Method

### 5.1 Baseline Model Used for Importance Extraction

Both notebooks trained an XGBoost classifier as a baseline model for feature importance extraction. The model configuration was:

| Parameter | Value |
| --- | --- |
| Model | `XGBClassifier` |
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `eval_metric` | `logloss` |
| `random_state` | 42 |

The training accuracy was used only as a sanity check to confirm that the model was learning from the feature matrix. It was not used as the final evaluation result for the research, because model evaluation is a separate later stage.

Observed training accuracy:

| Dataset | Training Accuracy |
| --- | ---: |
| URL phishing dataset | 0.9846 |
| Email phishing dataset | 0.9911 |

These high training accuracies suggest that the selected feature spaces contain strong predictive signals. However, because these are training-set results, they should not be interpreted as final model performance.

### 5.2 Primary Ranking Criterion: XGBoost Gain

The primary ranking method was XGBoost gain-based feature importance:

```python
model.get_booster().get_score(importance_type='gain')
```

Gain measures the average improvement in the model's objective function produced by splits using a given feature. A feature with a higher gain score contributed more to reducing classification error during tree construction. In this project, the gain ranking was used to select the final top-7 features because it directly reflects how strongly each feature contributed to the trained XGBoost model.

### 5.3 Secondary Interpretability Check: SHAP

SHAP TreeExplainer was used as a secondary interpretability method. For each dataset, SHAP values were computed on the held-out test set and ranked using mean absolute SHAP value:

```python
abs(shap_values).mean(axis=0)
```

This check was used to compare the gain-based top-7 features with the SHAP top-7 features. The purpose was not to replace the gain ranking, but to identify whether the same features remained influential under a different explanation method.

## 6. URL Feature Importance Results

### 6.1 Gain-Based Ranking

The full URL ranking was saved to:

```text
outputs/url_feature_importance/feature_importance_ranked.csv
```

The top-10 gain-ranked URL features were:

| Rank | Feature | Gain Score |
| ---: | --- | ---: |
| 1 | `HTTPS` | 70.381775 |
| 2 | `AnchorURL` | 26.746372 |
| 3 | `PrefixSuffix-` | 19.927757 |
| 4 | `ServerFormHandler` | 5.216730 |
| 5 | `WebsiteTraffic` | 4.156595 |
| 6 | `GoogleIndex` | 2.941844 |
| 7 | `DNSRecording` | 2.832114 |
| 8 | `LinksInScriptTags` | 2.744078 |
| 9 | `SubDomains` | 2.351866 |
| 10 | `DisableRightClick` | 2.348221 |

### 6.2 Selected Top-7 URL Features

The final selected URL feature subset was saved to:

```text
outputs/url_feature_importance/selected_features_top7.txt
outputs/url_feature_importance/selected_features_top7.json
```

The selected URL features are:

| Rank | Selected Feature | Gain Score | Interpretation |
| ---: | --- | ---: | --- |
| 1 | `HTTPS` | 70.381775 | The presence, absence, or trustworthiness of HTTPS-related behavior was the strongest URL signal. |
| 2 | `AnchorURL` | 26.746372 | Anchor-link behavior was highly informative, likely because phishing pages often use suspicious or mismatched link targets. |
| 3 | `PrefixSuffix-` | 19.927757 | Hyphenated domain patterns were important, which aligns with common phishing tactics that imitate legitimate domains. |
| 4 | `ServerFormHandler` | 5.216730 | Form submission handling was important, reflecting how phishing pages collect credentials or user input. |
| 5 | `WebsiteTraffic` | 4.156595 | Traffic/popularity information helped distinguish legitimate sites from suspicious or low-reputation pages. |
| 6 | `GoogleIndex` | 2.941844 | Search engine indexing status contributed to detection, as phishing pages may be less likely to be indexed. |
| 7 | `DNSRecording` | 2.832114 | DNS record properties were useful, reflecting domain registration and hosting characteristics. |

### 6.3 URL SHAP Consistency Check

The URL SHAP beeswarm plot was saved to:

```text
outputs/url_feature_importance/shap_summary_beeswarm.png
```

The notebook reported a SHAP top-7 versus gain-based top-7 overlap of `5/7` features. The discrepancy set was:

```text
DNSRecording, GoogleIndex, LinksInScriptTags, SubDomains
```

This means five of the seven selected gain-based features also appeared among the top seven SHAP-ranked features. Two gain-selected features, `GoogleIndex` and `DNSRecording`, were not in the SHAP top-7; conversely, `LinksInScriptTags` and `SubDomains` appeared in the SHAP top-7 but not in the gain top-7.

The difference does not invalidate the gain-based selection. Instead, it shows that some middle-ranked URL features are close enough in explanatory strength that their ordering depends on the interpretation method. For reporting purposes, the gain-based top-7 remains the selected subset, while the SHAP comparison should be cited as evidence that the selection was checked using an additional interpretability method.

## 7. Email Feature Importance Results

### 7.1 Gain-Based Ranking

The full email ranking was saved to:

```text
outputs/email_feature_importance/email_feature_importance_ranked.csv
```

The top-10 gain-ranked email features were:

| Rank | Feature | Gain Score |
| ---: | --- | ---: |
| 1 | `body_length` | 54.348064 |
| 2 | `body_html_tag_count` | 53.646465 |
| 3 | `body_caps_ratio` | 27.596704 |
| 4 | `urls_present` | 24.533352 |
| 5 | `body_urgency_count` | 21.553209 |
| 6 | `body_link_count` | 20.940956 |
| 7 | `body_exclamation_count` | 20.255928 |
| 8 | `subject_urgency_count` | 12.717711 |
| 9 | `subject_exclamation_count` | 11.167539 |
| 10 | `domain_mismatch` | 8.794921 |

### 7.2 Selected Top-7 Email Features

The final selected email feature subset was saved to:

```text
outputs/email_feature_importance/email_selected_features_top7.txt
outputs/email_feature_importance/email_selected_features_top7.json
```

The selected email features are:

| Rank | Selected Feature | Gain Score | Interpretation |
| ---: | --- | ---: | --- |
| 1 | `body_length` | 54.348064 | The length of the email body was the strongest email-level signal. This may reflect structural differences between phishing/spam-like messages and legitimate messages. |
| 2 | `body_html_tag_count` | 53.646465 | HTML structure was highly influential, suggesting that markup-heavy messages are important in the dataset's phishing patterns. |
| 3 | `body_caps_ratio` | 27.596704 | The proportion of uppercase text in the email body contributed strongly, capturing aggressive or attention-seeking formatting. |
| 4 | `urls_present` | 24.533352 | URL presence/count was important, which is expected because phishing emails commonly direct users to external sites. |
| 5 | `body_urgency_count` | 21.553209 | Urgency-related language in the body was a strong behavioral cue. |
| 6 | `body_link_count` | 20.940956 | Explicit link count in the body was important, reinforcing the role of embedded links in phishing attempts. |
| 7 | `body_exclamation_count` | 20.255928 | Exclamation mark usage in the body contributed to the model, likely capturing persuasive or alarmist message style. |

### 7.3 Email SHAP Consistency Check

The email SHAP beeswarm plot was saved to:

```text
outputs/email_feature_importance/email_shap_summary_beeswarm.png
```

The notebook reported a SHAP top-7 versus gain-based top-7 overlap of `5/7` features. The discrepancy set was:

```text
body_html_tag_count, body_urgency_count, subject_caps_ratio, subject_length
```

This means five of the seven selected gain-based email features also appeared among the top seven SHAP-ranked features. Two gain-selected features, `body_html_tag_count` and `body_urgency_count`, were not in the SHAP top-7; conversely, `subject_length` and `subject_caps_ratio` appeared in the SHAP top-7 but not in the gain top-7.

This suggests that the body-level features dominate the gain-based XGBoost ranking, while SHAP assigns relatively stronger explanatory weight to some subject-line characteristics. The finding is useful for discussion because it shows that email phishing cues are distributed across both body content and subject metadata, even though the final selected gain-based subset is body-heavy.

## 8. Summary of Selected Feature Sets

The final top-7 features selected for the next stage are:

| Dataset | Selected Features |
| --- | --- |
| URL phishing | `HTTPS`, `AnchorURL`, `PrefixSuffix-`, `ServerFormHandler`, `WebsiteTraffic`, `GoogleIndex`, `DNSRecording` |
| Email phishing | `body_length`, `body_html_tag_count`, `body_caps_ratio`, `urls_present`, `body_urgency_count`, `body_link_count`, `body_exclamation_count` |

These selected feature sets should be used as inputs for the next phase of the project: model training.

## 9. Statement of Findings

The feature importance extraction stage produced the following findings:

1. The URL phishing dataset contained no missing values after preprocessing and had a usable, moderately balanced binary target distribution.
2. The email phishing dataset required feature engineering before model-based ranking because its raw fields were textual and metadata-based.
3. XGBoost gain importance identified seven dominant URL features and seven dominant email features suitable for downstream model training.
4. For URL phishing detection, the most important signals were security, hyperlink, domain-pattern, form-handling, traffic, indexing, and DNS-related indicators.
5. `HTTPS` was the strongest URL feature by a wide margin, with a gain score of `70.381775`.
6. `AnchorURL` and `PrefixSuffix-` were also highly important URL features, indicating that link behavior and domain naming patterns are central to URL phishing detection in this dataset.
7. For email phishing detection, the most important signals were mostly body-level structural and stylistic indicators.
8. `body_length` and `body_html_tag_count` were the strongest email features, with gain scores of `54.348064` and `53.646465` respectively.
9. URL and link-related behavior remained important in the email dataset through `urls_present` and `body_link_count`.
10. Urgency and attention-seeking writing style were also useful email phishing indicators, represented by `body_urgency_count`, `body_caps_ratio`, and `body_exclamation_count`.
11. SHAP validation showed a `5/7` top-feature overlap with gain-based selection for both datasets.
12. The SHAP comparison supports the general reliability of the selected features while also identifying features that should be discussed as ranking-sensitive.
13. The final selected top-7 feature lists are sufficiently compact and interpretable for the next model training phase.

## 10. Implications for the Model Training Phase

The next phase should train candidate models using the selected feature subsets rather than the full original feature spaces. This will allow the research to test whether a smaller, interpretable feature set can still produce strong phishing detection performance.

Recommended next steps:

1. Build a training pipeline that loads the selected top-7 feature lists from the saved JSON artifacts.
2. Train baseline and comparative models separately for the URL and email datasets.
3. Use stratified train/test splitting with the same `random_state=42` for reproducibility unless a cross-validation design is adopted.
4. Compare model performance using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC where appropriate.
5. Pay particular attention to recall for the phishing class, because false negatives may be more harmful in a phishing detection context.
6. In the final report, clearly distinguish the feature importance model from the final evaluated detection model.

## 11. Generated Artifacts

The following artifacts were produced by this stage:

| Artifact | Purpose |
| --- | --- |
| `outputs/url_feature_importance/feature_importance_ranked.csv` | Full gain-based URL feature ranking. |
| `outputs/url_feature_importance/selected_features_top7.txt` | Plain-text selected URL feature list. |
| `outputs/url_feature_importance/selected_features_top7.json` | JSON selected URL feature list for downstream code. |
| `outputs/url_feature_importance/feature_importance_top10.png` | URL top-10 gain importance bar chart. |
| `outputs/url_feature_importance/shap_summary_beeswarm.png` | URL SHAP beeswarm interpretation plot. |
| `outputs/email_feature_importance/email_feature_importance_ranked.csv` | Full gain-based email feature ranking. |
| `outputs/email_feature_importance/email_selected_features_top7.txt` | Plain-text selected email feature list. |
| `outputs/email_feature_importance/email_selected_features_top7.json` | JSON selected email feature list for downstream code. |
| `outputs/email_feature_importance/email_feature_importance_all.png` | Email gain importance bar chart. |
| `outputs/email_feature_importance/email_shap_summary_beeswarm.png` | Email SHAP beeswarm interpretation plot. |

## 12. Conclusion

This stage successfully completed the first project goal by extracting the 5-7 most important features for the URL and email phishing detection tasks. The process was data-driven, reproducible, and supported by both XGBoost gain importance and SHAP-based interpretability checks. The selected feature sets provide a strong foundation for the next stage of the research, where machine learning models will be trained and later evaluated using formal classification metrics.
