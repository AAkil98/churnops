# Business & ML Problem Spec

## 1.1 Domain & Context

Subscription-based (telecom provider) is experiencing customer churm. Each customer has a set of account attributes, service subscriptions, and billing information. The business currently has no systematic way to predict which customers are likely to leave, making retention efforts reactive and untargeted.

## 1.2 Business Objective

Enable the marketing/retention team to **proactively identify high-risk customers** and intervene with targeted retention offers *before* they churn.

Success criteria from a business standpoint:

- Increase retention rate by at least 5% through proactive intervention on the top 2-% most-at-risk customers.
- Reduce unnecessary disconfigured by not targeting customers who would stay anyway.

## 1.3 ML Task Formulation

**Task:** Binary classification

| **Element** | **Definition** |
|-------------|----------------|
| **Input** | Customer demographic, account, and service features (snapshot at time T) |
| **Output** | Probability that the customer will churn within a defined future window |
| **Positive class** | Churn = Yes |
| **Negative class** | Churn = No |

The dataset provides a `churn` label (yes/no) for a given historical period. We treat this as a point-in-time classification problem.

## 1.4 ML Success Metrics

| **Metric** | **Definition** | **Target** |
|------------|----------------|------------|
| **ROC-AUC** | Measures ranking threshold quality across thresholds | > 0.80 |
| **Precision @K** (K=top 20%) | How many flagged as high risk actually churn | > 0.60 |
| **Recall @K** (K=top 20%) | How many actual churners we capture in top 20% | > 0.70 |
| **F1-Score** | Harmonic mean at optimal threshold | > 0.55 |

Justification:

- AUC ensures overall model discrimination
- Precision/Recall@K aligns with the business plan: target the top 20% highest-risk customers. We care about capturing churners (recall) without wasting too many offers on non-churners (precision).

## 1.5 Inference Constraints

| **Constraint** | **Value** |
|----------------|-----------|
| **Mode** | Batch (daily scoring of full customer base) |
| **Latency** | Not real-time; batch completion within 5 minutes ~ 7,000 customers |
| **Throughput** | ~7,000 predictions/day comfortably (CPU only) |
| **Cost** | Zero infrastructure cost (local deployment demo) |

Also: a REST API endpoint will be built for *demo purposes* (single customer scoring), not for production throughput.


## 1.6 Explainability Requirement

The retention team needs to know *why* a customer is flagged. For each high-risk prediction, we will surface:
- Top 3 most influential features (via SHAP values)
- Direction of influence (e.g., "Month-to-month contract → higher churn risk")

## 1.7 Offline vs. Online Evaluation

Since this is a portfolio project with no live business:

- **Offline evaluation** only, using held-out test set and time-based splits.
- **Simulated online evaluation:** We'll replay historical data as a "new batch" to demonstrate monitoring, drift selection, and retriggering logic.

## 1.8 Out of Scope (v1)

- Multi-class churn (voluntary vs. involuntary)
- Customer lifetime value integration
- Real-time streaming inference
- A/B experminetation framework (only shadow deployment simulation)

---

## 2. Data

### 2.1 Data Source

We'll use the publicly available **Telco Customer Churn** dataset (Kaggle/IBM). It contains 7,043 customers with 20 features and a binary churn label.

**Source file:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 2.2 Schema

| **#** | **Feature** | **Type** | **Description** |
|-------|-------------|----------|-----------------|
| 1 | `customerID` | string | Unique identifier |
| 2 | `gender` | categorical | Male / Female |
| 3 | `SeniorCitizen` | binary int | 0 = No, 1 = Yes |
| 4 | `Partner` | categorical | Yes / No |
| 5 | `Dependents` | categorical | Yes / No |
| 6 | `tenure` | numeric | Months with company |
| 7 | `PhoneService` | categorical | Yes / No |
| 8 | `MultipleLines` | categorical | Yes / No / No phone service |
| 9 | `InternetService` | categorical | DSL / Fiber optic / No |
| 10 | `OnlineSecturity` | categorical | Yes / No / No internet service |
| 11 | `OnlineBackup` | categorical | Yes / No / No internet service |
| 12 | `DeviceProtection` | categorical | Yes / No / No internet service |
| 13 | `TechSupport` | categorical | Yes / No / No internet service |
| 14 | `StreamingTV` | categorical | Yes / No / No internet service |
| 15 | `StreamingMovies` | categorical | Yes / No / No internet service |
| 16 | `Contract` | categorical | Month-to-month / One year / Two year |
| 17 | `PaperlessBilling` | categorical | Yes / No |
| 18 | `PaymentMethod` | categorical | Electonic check / Mai check / Bank transer / Credit card |
| 19 | `MonthlyCharges` | numeric float | Monthly amount charged |
| 20 | `TotalCharges` | numeric (string) | Total amount charged date |
| 21 | `Churn` | categorical | Yes / No (target) |

### 2.3 Data Volume

| **Stat** | **Value** |
|----------|-----------|
| Total rows | 7,043 |
| Positive class (Churn=Yes) | ~1,869 (26.5%) |
| Negative class (Churn=No) | ~5,174 (73.5%) |
| Class ratio | ~2.81:1 (moderate imbalance) |

This is comfortablity CPU-trainable. No sampling or distributed processing needed.

### 2.4 Known Data Issues to Handle in Pipeline

1. `TotalCharges` has blank strings for customers with `tenure=0` (new customers). Must coerce to numeric, replace blanks with 0 (or `MonthlyCharges`).
2. `customerID` is not a feature — drop before training
3. Code-mixing: `SeniorCitizen` is already 0/1, but `Partner`, `Dependents`, etc. Are "Yes"/"No" strings. Need consistent encoding.
4. Target `Churn` is "Yes"/"No" → map to 1/0.
5. Categorical features with `No internet service` and `No phone service` values — these encode a dependency on `InternetService`/`PhoneService`. May collapse or handle carefully.
6. Mild class imbalance — handle via `scale_pos_weight` in XGBoost or SMOTE if needed.

### 2.5 Train/Validation/Test Split Strategy

**Temporal constraint:** The dataset has no timestamp. We simulate a production-like split:

| **Split** | **%** | **Purpose** |
|-----------|-------|-------------|
| **Train** | 60% | Model training + hyperparameter tuning |
| **Validation** | 20% | Threshold selection, model comparison |
| **Test** | 20% | Final held-out evaluation (simulates "future" data) |

**Method:** Stratified random split by `Churn` (preserve class ratio across splits).

For drift simulation later:

- We'll hold back a small "deployment" subset from Test as "incoming batch 2" with artificially injected minor drift.
- Train/Val/Test = 60/20/20 of available data; deployment batch synthesized from Test.

### 2.6 Data Versioning Strategy

**Tool:** DVC (Data Version Control)

| **Element** | **Approach** |
|-------------|--------------|
| Raw data | Tracked by DVC, stored in `data/raw/` |
| Processed data | `data/processed` — output of preprocessing step, DVC-tracked |
| Splits | `data/splits/` — train.csv, val.csv, test.csv, DVC-tracked |
| Remote storage | Local directory simulated as remote (`data/remote`) for DVC push/pull demo |

Every pipeline run logs the DVC commit hash of the input data, linked in MLflow expriment metadata.

### 2.7 Feature Store

**V1: No standalone feature store.**
Features are computed on-the-fly in the preprocessing step. Feature definitions are versioned via the preprocessing code + DVC pipeline stage.
If we extend to v2, we'd mock a simple Feat setup for point-in-time lookups.

### 2.8 Expected Preprocessing Output

After preprocessing. the dataset for training will have:

- **Numerical features:** `tenure`, `MonthlyCharges`, `TotalCharges` (scaled/standardized)
- **One-hot encoded categoricals** (all string columns except `customerID`)
- **Target:** `churn` (int, 0/1)
- **No missing values**

Approximate dimensionality post-encoding: ~47 columns.

---

## 3. Model

### 3.1 Candidate Algorithms

Given tabular data, moderate size, and CPU-only constraint, three algorithms will be benchmarked:

| **Algorithm** | **Rationale** |
|---------------|---------------|
| **XGBoost** | State-of-the-art for tabular data, handles missing values, built-in regularization, scale_pos_weight for class imbalance |
| **LightGBM** | Faster training than XGBoost on CPU, often comparable performance, competetive benchmark |
| **Logistic Regression** | Baseline model, highly interpretable, fast to train, useful for calibration reference |

No deep learning — overkill and underperforms on tabular data of this size.

### 3.2 Primary Model Selection

**Primary: XGBoost**

Justification:

- Mature ecosystem, excellent MLFlow integration (built-in autologging)
- `scale_pos_weight` handles imabalance natively without data augmentation
- SHAP integration is seamless for explainability requirements
- Training time < 30 seconds on CPU for this dataset.

LightGBM kept as challenger. If it significantly outperforms XGBoost on validation metrics (>2% AUC improvement), it becomes the candidate for promotion.

Logistic Regression serves as the chanity-check baseline.

### 3.3 Hyperparameter Tuning Strategy

**Tool:** Optuna (with MLFlow callback for trial logging)

**Search space for XGBoost:**

| **Parameters** | **Range** | **Type** |
|----------------|-----------|----------|
| `n_estimators` | 100-500 | int |
| `max_depth` | 3-10 | int |
| `learning_rate` | 0.01-0.3 | float (log) |
| `subsample` | 0.6-1.0 | float |
| `colsample_bytree` | 0.6-1.0 | float |
| `min_child_weight` | 1-10 | int |
| `gamma` | 0-5 | float |
| `reg_alpha` | 1e-8-1.0 | float (log) |
| `reg_lambda` | 1e-8-1.0 | float (log) |
| `scale_pos_weight` | value derived from class ratio, fixed | fixed |

**Budget:** 50 trials, 5-fold cross-validatio on training set.
**Objective:** Maximize ROC-AUC (primary), F1 tracked as secondary.
**Pruning:** Median pruner, n_startup_trials=10.

### 3.4 Validation Strategy

| **Stage** | **Method** | **Purpose** |
|-----------|------------|-------------|
| **Hyperparameter tuning** | 5-fold stratified CV on train set | Select best hyperparams |
| **Model comparison** | Trail on full Train set, evaluate on Validation set | Compare XGBoost vs LightGBM vs Logitstic Regression |
| **Final evaluation** | Train on Train+Val, evaluate on held-out Test set | Unbiased performance estimate for model card |
| **Threshold selection** | Precision-Recall curve on Validation set | Choose threshold maximizing F1 for top-20% targeting |

### 3.5 Target Metrics for Model Acceptance

A model is elligible for **registration in MLflow Registry (Staging) if:**

| **Metric** | **Threshold** | **Evaluated On** |
|------------|---------------|------------------|
| ROC-AUC | > 0.82 | Test set |
| Precision@20% | > 0.60 | Test set |
| Recall@20% | > 0.68 | Test set |
| F1-score | > 0.55 | Test set |

If any metric falls below threshold, model is rejected from promotion. All metrics are automatically checked in the evaluatio pipeline stage.

### 3.6 Explainability Strategy

**Tool:** SHAP (TreeExplainer for XGBoost)

**Outputs:**

1. **Global:** SHAP summary bar plot (top 15 features) — saved as artifact for model card.
2. **Local (per prediction):** For REST API demo, return top 3 SHAP values with direction

SHAP values computed on a representative background sample (200 isntances from Train set) for efficieny on CPU.

### 3.7 Model Artifact

**Format:** MLFlow's native XSGBoost (`mlflow.xgboost`)

**What gets logged alongside the model:**

- Fitted pipeline object (preprocessing + model)
- Feature names and order
- SHAP explainer object
- Evaluated metrics (AUC, precision@20%, recall@20%, F1)
- Training data DVC commit hash
- Package requirements (`requirements.txt` snapshot)
- Input schema signature (MLflow `infer_signature`)

**Artifact name in registry:** `churn-predictor`

### 3.8 Reproducibility

A single config file `configs/model_config.yaml` pins:

- Algorithm choice
- Hyperparameter search space
- Random seed (42)
- Cross-validation folds
- Metric thresholds

Training is fully reproducible from config + data version.

---

## 4. Experiment Tracking

### 4.1 Tool Selection

**Tool:** MLflow

**Rationale:**

- Native integration with XGBoost, Optuna, SHAP
- Built-in model registry
- Local tracking server sufficient for CPU-only single-machine workflow
- Python-native, no external dependency like a database
- Strong portfolio signal (industry standard)

**Tracking server:** Local instance, URI = `http://127.0.0.1:5000`
**Backend store:** SQLite (`mlruns/mlflow.db`)
**Artifact store:** Local filesystem (`mlruns/artifacts/`)

No remote server needed. All runs stay local.

### 4.2 What Gets Logged

Per experiment run, the following are automatically or manually logged:

| **Category** | **Items** | **Method** |
|--------------|-----------|------------|
| **Parameters** | All hyperparameters (from Optuna trial or fixed config) | `mlflow.log_params()` / Optuna MLflowCallback |
| **Metrics** | ROC-AUC, Precision@20%, Recall@20%, F1 (Trail, val, test) | `mlflow.log_metrics()` |
| **Tags** | `data_version` (DVC hash), `git_commit`, `expriment_type`, `algorithm` | `mlflow.set_tags()` |
| **Artifacts** | Model pickle, SHAP summary plot, confusion matrix, classification report, feature importance CSV, `requirements.txt` snapshot | `mlflow.log_artifacts()` / `mlflow.log_figure()` |
| **Model** | Full MLflow model (preprocessing pipeline + trained model) with signature | `mlflow.xgboost.log_model()` |
| **Dataset** | Schema signature inferred from a batch of training data | `mlflow.infer_signature()` |

### 4.3 Experiment Organization

**Experiment names** (created in MLflow UI or via code):

| **Experiment name** | **Purpose** |
|---------------------|-------------|
| `churnops-baseline` | Logistic regression baseline runs |
| `churnops-xgboost-tuning` | All Optuna hyperparameter search trials |
| `churnops-lightgbm-tuning` | LightGBM Optuna trials (challenger) |
| `churnops-final-train` | Final model training on Train+Val, evaluated on Test |
| `churnops-drift-simulation` | Runs triggered by drift detection simulation |

**Nested runs:** Optuna parent run contains child runs per trial

### 4.4 Run NAming Convention

**Pattern:** `<algorith>_<date>_<short-hash>`
**Example:** `xgboost_2026-05-04_a3f2`

Optuna trials: `trial_<numbers>` (auto-generated) under parent run.

### 4.5 Model Comparison & Promotion Criteria

From the MLflow UI, we compare:

- All finalist runs from `churnops-xgboost-tuning` and `churnops-lightgbm-tuning`
- Comparison view: metrics table + parallel coordinates plot

**Promotion decision logic** (automated in pipeline):

1. Select best run from each algoritm expriment by Validation ROC-AUC.
2. Evaluate both on Test set.
3. If XGBoost satisfies all acceptance thresholds and outperforms or ties LightGBM → promote XGBoost to **Staging**.
4. If LightGBM beats XGBoost by >2% in AUC on Test → flag for human review (logged as warning, manual promotion step)
5. Baseline (Logistic Regression) is never promoted; serves as lower bound.

### 4.6 MLflow Integration Points in Pipeline

| **Pipeline Stage** | **MLflow action** |
|--------------------|-------------------|
| Data preprocessing | Start of parent run: log `data_version` tag |
| Hyperparameter tuning | Optuna creates nested runs per trial unde parent |
| Model evaluation | Log test metrics and artifacts |
| Model registration | If metrics pass threshold: `mlflow.register_model()` with stage `Staging` |
| Model promotion | Manual (simulated) via MLflow UI: `Staging` → `Production` |

---

## 5. Pipeline & Orchestration

### 5.1 Orchestration Tool

**Tool:** Prefect (open-source, local execution)

**Rationale:**
- Python-native, decorated-based API — minimal boilerplate
- Local execution without need for a servet/agent for development
- Built-in retries, logging, caching, and scheduling
- Lightweight compared to Airflow for a single-machine portfolio project
- Increasing industry adoption, strong portfolio signal

**Execution:** Local, via `prefet flow run` or `python pipeline.py`. No prefect Cloud or server needed; Orion/Prefect server runs locally for UI.

### 5.2 Pipeline Architecture Overview

```text
┌─────────────┐
│ data/raw/   │
└─────────────┘
       │
       │
       ↓
┌────────────────┐
│ 1. Data ingest │ ← DVC pull (if needed), load CSV
└────────────────┘
       │
       │
       ↓
┌──────────────────────────┐
│ 2. Data validation       │ ← Schema check, missing values, outliers, drift vs reference
└──────────────────────────┘
       │
       │
       ↓
┌──────────────────────────┐
│ 3. Data Preprocessing    │ ← Encode, scale, split, save artifacts
└──────────────────────────┘
       │
       │
       ↓
┌────────────┐
│4. Training │ ← Test set merics, SHAP, thresholding
└────────────┘
      │
      │
      ↓
┌────────────────┐
│ 5. Evaluation   │ ← Test set Metrics, SHAP, thresholding
└────────────────┘
      │
      │
      ↓
┌─────────────────────────────┐
│ 6. Model Registry Updated   │ ← Promote to Staging if threshlds met
└─────────────────────────────┘
```

### 5.3 Detailed Pipeline Steps

**Step 1: Data ingestion**

| **Attribute** | **Detail** |
|---------------|------------|
| **Function** | `ingest_data()` |
| **Input** | Raw CSV (`data/raw/...`) |
| **Output** | Raw pandas DataFrame, logged as DVC-tracked artifact |
| **Actions** | Load CSV, verify row count, log DVC hash as Prefect Artifact |
| **Fails if** | FIle missing, row count below expected minimum (5000) |

**Step 2: Data Validation**

| **Attribute** | **Detail** |
|---------------|------------|
| **Function** | `validate_data()` |
| **Input** | RAw DataFrame |
| **Output** | Validation report (JSON), pass/fail flag |
| **Actions** | Check column presence, dtypes, null % (max allowed: 5% per col), class balance drift vs reference, basic outlier detection (IQR for tenure, MonthlyCharges) |
| **Fails if** | Missing critical columns, null % > threshold, catastrophic drift detected |
| **Warns if** | Minor drift detected, outliers present (pipeline continues) |
| **Tool** | Great expectations (optional, v2) or custom checks in v1 |

**Step 3: Data Preprocessing**

| **Attribute** | **Detail** |
|---------------|------------|
| **Function** | `preprocess_data()` |
| **Input** | Validated DataFrame |
| **Output** | `X_train, X_val, X_test, y_train, y_val, y_test` + fitted preprocessor object saved to disk |
| **Actions** | Parse `TotalCharges` → numeric, encode categoricals (OneHotEncoder), scale numerics (StandardScaler), stratified split (60/20/20), save splits to `data/splits/`, save fitted preprocessor as `.pkl` |
| **Artifacts** | Preprocessor pickle, split CSVs (DVC-tracked) |

**Step 4: Training**

| **Attribute** | **Detail** |
|---------------|------------|
| **Function** | `train_model()` |
| **Input** | Train + Val splits |
| **Output** | Best model, logged to MLflow |
| **Actions** | Start MLflow parent run, run Optuna (50 trials, -fold CV), select best trial by VAL AUC, restrain best on full Train set, log model + params + metrics to MLflow |
| **Artifacts** | MLflow run ID |

**Step 5: Evaluation**

| **Attribute** | **Detail** |
|---------------|------------|
| **Function** | `evaluate_model()` |
| **Input** | Best model run ID, Test split |
| **Output** | Test metrics, SHAP summary plot, confusion matrix, classification report |
| **Actions** | Load model from MLfow, score Test set, compute all threshold metrics (Precision@20%, Recall@20%, F1), generate SHAP plot, log everything to same MLflow run |
| **Artifacts** | Metrics JSON, SHAP plot PNG, confusion matrix PNG |

**Step 6: Model Registry Update**

| **Attribute** | **Detail** |
| **Function** | `register_or_reject()` |
| **Input** | Test metrics, MLflow run ID |
| **Output** | Model version in MLflow Registry (Staging) OR rejection log |
| **Actions** | Check all metrics against acceptance criteria thresholds, if pass: register model as `churn-predictor` with stage `Staging`, if fail: log reason, do not register |
| **Artifacts** | Registered model version (or rejection message) |

### 5.4 Pipeline Parameterization

All configurable values are externalized to a single YAML file: `configs/pipeline_config.yaml`

```yaml
data:
  raw_path: "data/raw/yy.csv"
  min_rows: 5000
  max-null_pct: 0.05
  test_size: 0.2
  val_size: 0.2
  random_seed: 42

training:
  algorithm: "xgboost"
  opotuna_trials: 50
  cv_folds: 5
  metric_objective: "roc_auc"

evaluation:
  auc_threshold: 0.82
  precision_at_20_threshold: 0.60
  recall_at_20_threshold: 0.68
  f1_threshold: 0.55

mflow:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "churnops-xgboost-tuning"
  registry_model_name: "churn-predictor"
```

### 5.5 Scheduling & Triggering

**Manual trigger (primary):**
- Developer runs: `python pipeline.py` or `prefet flow run`

**Automated retraining trigger (simulated for portfolio):**

- A second flow `trigger_retrain.py` watches for:
  - Manual "new data arrival: simulation (drop a new CSV in `data/incoming/`)
  - Data drift detected in monitoring flow
- On trigger, runs full pipeline end-to-end
- If new model passes thresholds → auto-register to Staging.

### 5.6 Failure Handling & Retries

| **Mechanism** | **Detail** |
|---------------|------------|
| **Retries** | Prefect `@flow(retries=2, retry_delay_seconds=30)` on Data Ingest and Training steps |
| **Cache** | Prefect task caching on Data Preprocessing (if data version unchanged, skip reprocessing) |
| **Notifications** | Log to console + write to `logs/pipeline.log`; no email/Slack (local project) |

### 5.7 Caching Strategy

| **Task** | **Cache Key** |
|----------|---------------|
| Data Ingestion | DVC Hash of raw data file |
| Data Validation | Hash of raw DataFrame |
| Data Preprocessing | Hash of validated DataFrame + config params |
| Training | Disabled (always retrain with Optuna; reproducibility via seed) |
| Evaluation | Run ID |

Prefect's built-in `cache_key_fn` or manual check via DVC hashes.

---

## 6. Model Registry & Versioning

### 6.1 Tool

**MLflow Model Registry** — built into the MLflow tracking server already specified.

No additional tooling required. The registry is accessible via:

- MLflow UI (`http://127.0.0.1:5000`)
- MLflow Python API (`mlflog.register_model()`, `mlflow.client.MlflowClient`)

### 6.2 Registered Model Name

`churn-predictor`
Single registered model. All versions (v1, v2, ...) live under this name. If we later add a challenger model type (LightGBM), it will be registered separately as `churn-predictor-lgb` and compared manually before any swap.

### 6.3 Model Stage & Lifecycle

| **Stage** | **Meaning** | **How it Gets There** |
|-----------|-------------|-----------------------|
| **None** | Newly registered, not assigned | Automatic upon `mlflow.register_model()` |
| **Staging** | Candidate for production, passed all metric tresholds | Set by pipeline (Step 6) automatically if Test metrics pass |
| **Production** | Currently deployed model, serving predictions | Manual promotion in MLflow UI (simulates approval gates) |
| **Archived** | Retired model, no longer used | Manual transition when a newer version is promoted to production |

**Lifecycle flow:**
```text
None ─(auto if metrics pass)─→ Staging ─(manual approve)─→ Production
                                  │                             │
                                  ↓                             ↓
                              (rejected)                      Archived
                              stays Staging or                (when replaced)
                              demoted to None
```

### 6.4 Version Numbering

Mlflow auto-increments integer versions: **Version 1, Version 2, Version 3...**

Each version corresponds to one training pipeline run that passed the evaluation gate.

### 6.5 Metadata Tracked Per Model Version

| **Metadata** | **Source** | **Example** |
|--------------|------------|-------------|
| **Version number** | Mlflow auto | `3` |
| **Stage** | MLflow Registry | `Production` |
| **Run ID** | Training run | `a3f2bc...` |
| **Metrics** | Logged in run | AUC=0.85, F1=0.58 |
| **Data version** | Tag on run | `dvc_hash=abc123` |
| **Git commit** | Tag on run | `git_commit=8d7f1e2` |
| **Algorithm + params** | Logged in run | XGBoost, max_depth=5, ... |
| **Training date** | Run timestamp | `2026-05-04 14:22:10` |
| **Promoted by** | MLflow user | `local_user` |

All metadata is viewable in the MLflow Registry UI per version.

### 6.6 Model Aliases (Optional Enhancement)

In addition to stages, we may assign an alias:

| **Alias** | **Points To** | **Purpose** |
|-----------|---------------|-------------|
| `champion` | Current production version | Used by serving code to load the deployed model |
| `challenger` | Latest Staging Version | Used in  shadow deployment simulation for comparison |

Aliases are set via `MLflowClient.set_registered_model_alias()`. This decouples deployment from the stage string.

### 6.7 Promotion Approval Simulation

In a real organization, promotion from Staging → Production requires a human approval step. We simulate this:

1. Pipeline auto-sets new model to **Staging** if metrics pass.
2. A script `scripts/promote_to_production.py` reads the latest Staging version and:
  - Print a summary of metrics vs current Production version (if exists)
  - Prompts user: `Promote version X to Production? [y/N]`
3. On confirmation, transitions the model to Production and archives the previous Production version.

This mimics a CI/CD approval gate without external tooling.

### 6.8 Serving Load Logic

The inference service loads the model by alias:

```python
model = mlflow.pyfunc.load_model("models:/churn-predictor@champion")
```

If no champion alias exists (first run), falls back to `models:/churn-predictor/Production`.

This ensures the serving container always picks up the correct model without hardcoding version numbers.

### 6.9 Archive Policy

| **Rule** | **Action** |
|----------|------------|
| Only one version in Production at a time | Previous Production → Archived |
| Staging can have multiple versions | Old staging versions retained for comparison; manually cleaned |
| Archived versions kept indefinitely | Small storage footprint, no need for deletion in portfolio |

### 6.10 Model Lineage Example

```text
churn-predictor
├─ Version 1 [Archived] (XGBoost, AUC=0.83, data_v1, initial model)
├─ Version 2 [Production] (XGBoost, AUC=0.85, data_v2, retrained)
├─ Version 3 [Staging] (XGBoost, AUC=0.87, data_v3, new challenger)
└─ Version 4 [None] (failed eval, not promoted)
```

---

## 7. Deployment

### 7.1 Deployment Modes

Two modes are implemented to demonstrate different serving patterns:

| **Mode** | **Purpose** | **Trigger** |
|----------|-------------|-------------|
| **Batch Inference** | Score entire customer base daily, output CSV | Scheduler/CLI |
| **Real-time REST API** | Score single customer on-demand (demo) | HTTP request |

### 7.2 Batcch Inference

#### Design

| **Attribute** | **Detail** |
|---------------|------------|
| **Script** | `scripts/batch_predict.py` |
| **Input** | CSV file of customers (same schema as training data minus `Churn` column) |
| **Output** | CSV with original columns + `churn_probability` + `predicted_churn` (at chosen threshold) + top 3 SHAP reasons as columns |
| **Model loading** | Loads from MLflow Registry by alias `champion` (or `Production` stage) |
| **Execution** | CLI: `python scripts/batch_predict.py --input data/incoming/batch.csv --output data/predictions/batch.csv` |

#### Artifacts

- Prediction CSV saved to `data/predictions/`
- SHAP explanations embedded in output columns
- Summary statistics printed to console (churn rate %, top risk factors)

### 7.3 Real-Time REST API

#### Stack

| **Component** | **Technology** |
| **Framework** | FastAPI |
| **Server** | Uvicorn |
| **Container** | Docker
| **Model loading** | MLflow `pyfunc` from registry at startup |

#### API Endpoints

| **Method** | **Path** | **Purpose** |
|------------|----------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "healthy", "model_version": 2}` |
| `POST` | `/predict` | Single customer prediction |
| `GET` | `/model-info` | Returns current model version, stage, metrics summary |

#### Request/Response Schema (`/predict`)

**Request:**

```json
{
  "customerID": "12234-ADB",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
}
```

**Response:**

```json
{
  "customerID": "1111",
  "churn_probability": 0.78,
  "predicted_churn": true,
  "threshold": 0.45,
  "top_factors": [
    {"feature": "Contract_Month-to-moth", "impact": +0.22},
    {"feature": "tenure", "impact": -0.15},
    {"feature": "MonthlyCharges", "impact": +0.11}
  ],
  "model_version": 2
}
```

#### API Startup Behavior

1. On boot, connects to MLflow tracking server
2. Loads model: `models:/churn-prediction@champion` (or `Production`)
3. Loads fitted preprocessor (bundled in MLflow model artifact)
4. Loads SHAP explainer (bundled in artifact)
5. Keeps model in memory; no reload needed unless service restarts.

#### Error Handling

| **Scenario** | **HTTP Code** | **Response** |
|--------------|---------------|--------------|
| Missing required field | 422 | Pydantic validation error detail |
| Invalid value type | 422 | Field-level error |
| Model not loaded | 503 | `{"error": "Model not available"}` |
| MLflow unreachable at startup | Fatal exit | Logged to stderr |

### 7.4 Containerization

#### Docker Image

| **Attribute** | **Detail** |
|---------------|------------|
| **Best image** | `python:3.10-slim` |
| **Dependencies** | Installed via `requirements.txt` (generated from Poetry/pip-compile) |
| **MLflow URI** | Passed as env var `MLFLOW_TRACKING_URI` |
| **Port** | 8000 exposed |
| **Command** | `uvicorn app.main:app --host 0.0.0.0 --port 8000` |

#### Dockerfile Structure

```text
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY configs/ ./configs/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.5 Local Production Simulation

**Docker Compose** file orchestrates the full local stack:

| **Service** | **Purpose** |
|-------------|-------------|
| `mlflow-server` | MLflow Tracking + Registry UI on port 5000 |
| `churn-api` | FastAPI inference service on port 8000 |
| *Optional* `streamlit-dashboard` | Monitoring dashboard on port 8501 |

A single `docker-compose up` launches the entire "production" environment locally.

### 7.6 Model Reload Strategy

For the portfolio demo, model reload is **manual**:

- Retrain pipeline runs
- If new model promoted to Production, restart the API container to pick it up.
- Simulated zero-downtime by having Docker Compose restart with `docker-compose up -d churn-api`.

No live reload watcher in v1.

### 7.7 API Testing

| **Test Type** | **Tool** | **Scope** |
|---------------|----------|-----------|
| Unit tests | pytests | Preprocessing logic, schema validation |
| Integration tests | pytest + FastAPI Test Client | `/health`, `/predict` with sample payload, verify response shapre and values |
| Load test (optional) | locust | 10 req/s to `/predict`, verify p95 latency < 200ms |

---

## 8. CI/CD

### 8.1 Version Control & Hosting

| **Attribute** | **Detail** |
|---------------|------------|
| **Platform** | GitHub |
| **Repo name** | `churnops` |
| **Branching strategy** | Trunk-based: `main` is the single source of truth; feature breanches merged via PR |
| **Protection** | `main` requires passing CI checks before merge (simulated via local pre-commit + GitHub Actions) |

### 8.2 CI Pipeline (Continuous Integration)

**Trigger:** On every push to `main` and on every pull request to `main`.
**Tool:** GitHub Actions

#### CI Job Flow

```text
┌──────────────┐
│  1. Lint     │ (ruff, black --check)
└──────────────┘
      │
      │
      ↓
┌───────────────┐
│ 2. Type Check │ (mypy, optional v2)
└───────────────┘
      │
      │
      ↓
┌───────────────┐
│ 3. Unit tests │ (pytest on preprocessing, validation, API schemas)
└───────────────┘
      │
      │
      ↓
┌───────────────────┐
│ 4. Pipeline Smoke │ (mini end-to-end: ingest → preprocess → train on 100 rows → evaluates)
└───────────────────┘
```

#### CI Job Details

| **Job** | **Command** | **Pass Condition** |
|---------|-------------|--------------------|
| **Lint** | `ruff check .`, `black --check .` | Zero errors |
| **Unit tests** | `pytest tests/unit/ -v` | All tests pass |
| **Smoke Tess** | `python scrips/smoke_test_pipeline.py` | Completes without error, AUC > 0.5 (above random) |

#### CI Environment

- GitHub-hosted runner (`ubuntu-latest`)
- Python 3.10
- Dependencies from `requirements.text` (no MLflow server needed for unit tests; smoke test uses local temporary directory as MLflow backed)
- DVC not pulled (smoke test uses a tiny synthetic dataset checked into the repo)

### 8.3 CD Pipeline (Continuous Deployment / Delivery)

#### Trigger Options

Two triggers, both implemented as separata GitHub Actions workflows:

| **Trigger** | **Workflow File** | **Purpose** |
|-------------|-------------------|-------------|
| **Manual** | `retrain.yml` | Developed triggers via `worfklow_dispatch` (GitHub UI button) |
| **Scheduled** | `retrain-scheduler.yml` | Cron job simulating weekly retraining (or "new data arrived") |

#### CD Job Flow

```
┌──────────────────────┐
│ 1. Setup Environment │ (checkout, Python, deps, start MLflow if local)
└──────────────────────┘
      │
      │
      ↓
┌──────────────────────┐
│ 2. DVC Pull Data     │ (Pull latest data version)
└──────────────────────┘
      │
      │
      ↓
┌──────────────────────┐
│ 3. Run Full Pipeline │ (ingest → validate → preprocess → train → evaluate → register)
└──────────────────────┘
      │
      │
      ↓
┌──────────────────────┐
│ 4. Model Promotion   │
│  Decision            │
└──────────────────────┘
          │
          │
    ───────────────
    │             │
    │             │
    Pass          Fail
    │             │
    ↓             ↓
  Promote       Log + Alert (issue/comment on commut)
  To
  staging
    │
    ↓
┌───────────────────────┐
│ 5. Build Docker Image │ (only if promoted, tag with version)
└───────────────────────┘
      │
      │
      ↓
┌──────────────────────┐
│ 6. Smoke test deploy │ (starat container, hit /health, hit /predict)
└──────────────────────┘
```

#### CD Environment Constraints

Since this is local-first portfolio project:

- **Real CD to cloud is out of scope** (no AWS/GCP/Azure)
- GitHub Actions simulates the pipeline but **cannot run MLflow server + Docker compose natively** without a self-hosted runner
- **Solution:** The CD workflow in GitHub Actions runs the **training + evaluation + registry logic** only. Docker build and serving smoke test are documented as **local commands** to run post-retrain.

The repo will include:

- `scripts/cd_local.sh` — one-command script to: pull latest data, run pipeline, build Docker image, restart API container, smoke test.
- GitHub Actions workflow runs the subset that's CI-runnable; the full CD is demonstrated locally and documented via a demo script/screen recording.

### 8.4 Promotion Approval Gate

| **Gate** | **Mechanism** |
|----------|---------------|
| **Auto-promoting to Staging** | Pipeline script checks metric thresholds and registers model |
| **Manual promotion to Production** | Developer runs `scripts/promote_to_production.py` |

This simulates a human-in-the-loop approval process common in regulated industries.

### 8.5 Artifact Management

| **Artifact** | **Storage** | **Versioning** |
|--------------|-------------|----------------|
| Data | DVC remote (local `data/remote/` dir, simulating S3) | DVC commit hash |
| Models | MLflow artifact store (`mlruns/artifacts/`) | MLflow model version |
| Docker images | Local Docker images | Tagged with model version: `churnops-api:v2` |
| Configs | Git | Git commit SHA |

### 8.6 Rollback Strategy

If a newly promoted model exhibits issues (simulated by checking API responses post-deploy):
1. Manually re-promote previous Production version in MLflow Registry
2. Rebuild Dcoker image with `champion` alias pointing to rolled-back version.
3. Restart API container

Documented as `scripts/rollback.sh`

### 8.7 CI/CD File Tree (Relevant to CI/CD)

```
.github/
  workflows/
    ci.yml          # lint + unit tests + smoke test on push/PR
    retrain.yml     # Manual trigger: run full training pipeline
    retrain-scheduled.yml # Cron trigger (weekly)

scripts/
  cd_local.sh       # Full local CD: pipeline + docket build + restart + smoke
  promote_to_production.py  # Manual promotion script
  rollback.sh       # Rollback to previous production version
  smoke_test_pipeline.py    # Mini pipeline for CI smoke test
```

---

## 9. Monitoring & Drift Detection

### 9.1 Monitoring Objectives

| **Objective** | **How** |
|---------------|---------|
| **Data drift** | Detect when incoming feature distributions deviate from training reference |
| **Model Performance Degradation** | Simulate label arrival later, compare predicted vs actual |
| **Prediction Drift** | Monitor prediction distribution shifts over time |
| **System health** | API uptime latency, error rate |

### 9.2 Tooling

| **Component** | **Tool** |
|---------------|----------|
| **Drift Detection** | Evidently AI (Open-source, compute PSI, KS-test, feature-level drift) |
| **Dashboard** | Streamlit (single-page monitoring overview) |
| **Logging/metrics** | JSON reports sved per batch, displayed in dashboard |
| **Alerting** | Logged warnings to file + console (simulated; no real alerting stack) |

### 9.3 Reference Data

The **Training dataset** (used to train the Production model) serves as the reference for all drift comparisons.

**Stored as:** `data/reference/reference_data.parquet`
**Includes:** feature distribution, target distribution, prediction distibution from training.

Generated and saved once at model training time (pipeline Step 5 side effect).

### 9.4 Drift Metrics

| **Metric** | **Applies To** | **Threshold** | **Action** |
|------------|----------------|---------------|------------|
| **PSI (Population Stability Index)** | Each feature | > 0.2 → warning, > 0.3 → Alert | Log, trigger retrain investigation |
| **KS-test p-value** | Each numeric feature | > 0.05 → drift flagged | Log feature name |
| **Chi-squared p-value** | Each categorical feature | > 0.05 → drift flagged | Log feature name |
| **Prediction drift (PSI)** | Predicted probabilities | > 0.15 → Warning, > 0.25 → Alert | Log, investigate |
| **Target drift** | Actual churn rate (if labels arrive later) | > 5 percentage point change | High priority alert |

### 9.5 Batch Monitoring Flow

Each batch prediction run triggers:

1. Run inference on new batch.
2. Load reference data from `data/reference/`.
3. Run Evidently **Data Drift Report** (new batch vs reference).
4. Run Evidently **Prediction Drift Report** (new predictions vs reference predictions).
5. Save repors as HTML to `data/monitoring/batch_YYYY-MM-DD/`.
6. Append summary metrics to `data/monitoring/drift_log.csv`.
7. If any threshold crosses → log WARNING/ALERT with details.

### 9.6 Delayed Label Simulation

In real churn predictions, labels (who actually churned) arrive weeks later. We simulate this:

1. Hold back a labeled test set as "future batch with labels".
2. A scrip `scripts/simulate_label_arrival.py`:
  - "Reveals" labels for a past batch.
  - Computes actual model performance (AUC, F1, precision/recall at threshold)
  - Compares to training-time test metrics.
  - Flags degradation if AUC drops > 0.05 from reference.

### 9.7 System Health Monitoring

| **Metric** | **Collection** | **Dashboard** |
|------------|----------------|---------------|
| API uptime | Manual / simulated | Status indicator (Green/red) |
| `/predict` latency | Logged per request (avg over last 100) | Time series line chart |
| Error rate | % of 4xx/5xx responses | Rolling bar chart |
| Prediction volume | Count per batch run | Bar chart |

### 9.8 Streamlit Dashboard Layout

Single page with sections:

```text
┌─────────────────────────────────────────┐
│ ChurnOps Monitoring Dashboard           │
│ Model: churn-predictor v2 (Production)  │
│ Last batch: 2026-05-05 08:00 UTC        │
├─────────────────────────────────────────┤
│ System Health                           │
│ [API Up] [Latency: 45ms p95] [0.2% err] │
├─────────────────────────────────────────┤
│ Data Drift Summary                      │
│ 3 features drifted (threshold > 0.2)    │
│   ┌─────────────────────────────────┐   │
│   │ Feature   │ PSI   │ Status      │   │
│   │ tenure    | 0.35  │ High Drift  │   │
│   │ monthly   | 0.22  │ Warning     │   │
│   │ contract  | 0.18  │ OK          │   │
│   └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│ Prediction Drift                        │
│ PSI: 0.12 │ OK                          │
├─────────────────────────────────────────┤
│ Drift Trend (PSI over time)             │
│ [Line chart: date vs max feature PSI]   │
├─────────────────────────────────────────┤
│ Recent Batch Performance (with labels)  │
│ [Bar chart: AUC per batch vs reference] │
└─────────────────────────────────────────┘
```

### 9.9 Alerting (Simulated)

| **Channel** | **How** |
|-------------|---------|
| Console | WARNING/ERROR log lines in pipeline output |
| Log file | `logs/monitoring.log` with structured JSON lines |
| Dashboard | Red indicators and alert banners in Streamlit |

No email/Slack/PagerDuty integration — out of scope for local portfolio.

### 9.10 Retraining Trigger Integration

When drift crosses threshold (any feature PSI > 0.3 or prediction PSI > 0.25):

- Monitoring script prints: `ALERT: Drift threshold excdeeded. Consider retraining.`
- A flag file `data/monitoring/retrain_needed.flag` is created
- The CD script can check for this flag and auto-trigger retraining

---

## 10. Reproducibility & Environment Spec

### 10.1 Dependency Management

| **Aspect** | **Tool** |
|------------|----------|
| **Python version** | 3.10 (pinned) |
| **Dependency resolver** | Poetry (`pyproject.toml`) |
| **Lock file** | `poetry.lock` committed to repo |
| **Export for Docker/CI** | `requirements.txt` auto-generated via `poetry export` |

### 10.2 Key Dependencies

```text
python = 3.10
pandas = 2.0
numpy = 1.24
scikit-learn = 1.3
xgboost = 2.0
lightgbm = 4.0
optuna = 3.4
mlflow = 2.8
prefect = 2.14
fastapi = 0.104
uvicorn = 0.24
evidently = 0.4
shap = 0.43
streamlit = 1.28
dvc = 3.40
pyyaml = 6.0
pytest = 7.4
ruff = 0.1
black = 23.0
```

### 10.3 Environment Isolation

| **Environment** | **Method** |
|-----------------|------------|
| Development | Poetry virtual env (`.venv/`) |
| CI (GitHub Actions) | `pip install -r requirements.txt` in fresh runner |
| Production (Docker) | `pip install --no-cache-dir -r requirements.txt` inside container |
| MLflow server | Same Docker Compose network as API |

### 10.4 Single-Command Reproducibility

A new developer (or reviewer) can reproduce eerything with:

```bash
#1. Clone repo
git clone https://github.com/AAkil98/churnops.git
cd churnops

# 2. Setup Environment
poetry install

# 3. Pull data
dvc pull

# 4. Run full pipeline
python pipeline.py

# 5. Launch production stack
docker-compose up -d

# 6. Verify
curl http://localhost:8000/health
```

### 10.5 Directory Structure

```text
churnops/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── retrain.yml
│       └── retrain-scheduled.yml
├── app/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Pydantic models
│   └── model_loader.py       # MLflow model loading logic
├── configs/
│   ├── pipeline_config.yaml
│   └── model_config.yaml
├── data/
│   ├── raw/                  # DVC-tracked raw data
│   ├── processed/            # DVC-tracked processed data
│   ├── splits/               # Train/val/test CSVs
│   ├── reference/            # Reference data for monitoring
│   ├── incoming/             # Simulated new batches
│   ├── predictions/          # Batch prediction outputs
│   ├── monitoring/           # Drift reports, logs
│   └── remote/               # Simulated DVC remote
├── mlruns/                   # MLflow artifacts (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb            # Initial data exploration
│   ├── 02_baseline_model.ipynb # Prototyping before code
│   ├── 03_model_comparison.ipynb  # Comparing models
│   └── 04_shap_analysis.ipynb  # Explainability deep dive
├── pipeline/
│   ├── ingest.py
│   ├── validate.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── register.py
├── scripts/
│   ├── batch_predict.py
│   ├── promote_to_production.py
│   ├── simulate_label_arrival.py
│   ├── cd_local.sh
│   ├── rollback.sh
│   └── smoke_test_pipeline.py
├── monitoring/
│   ├── drift_report.py       # Evidently report generation
│   └── dashboard.py          # Streamlit app
├── tests/
│   ├── units/
│   │   ├── test_preprocess.py
│   │   ├── test_validate.py
│   │   └── test_schemas.py
│   └── integration/
│       └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── poetry.lock
├── requirements.txt
├── .dvc/
├── .dvcignore
├── .gitignore
└── README.md
```

### 10.6 Seeds & Determinism

All randomness is seeded to `42` across:
- Trail/val/test split
- XGBoost `random_state`
- Optuna sampler

Full determinism not enforced (XGBoost non-determinstic by default on CPU), but results are highly reproducible with narrow variance.

---

## 11. Portfolio Packaging Spec

### 11.1 README Structure

```markdown
# ChurnOps: End-to-End MLOps for Churn Prediction

## Overview
1-2 paragraphs on problem, approach, tech stack.

## Architecture Diagram
(ASCII or PNG diagram showing full system)

## Tech Stack
- Data: DVC, pandas, scikit-learn
- Training: XGBoost, Optuna, MLflow
- Pipeline: Prefect
- Serving: FastAPI, Docker
- Monitoring: Evidently AI, Streamlit
- CI/CD: GitHub Actions

## Quick Start
5-6 commands to go from zero to running system.

## Project Structure
Directory tree with brief descriptions.

## Pipeline Walkthrough
Step-by-step with screenshots.

## API Usage
cURL example.

## Monitoring Dashboard
Screenshot + how to launch.

## CI/CD
How to trigger, what happens

## Design Decisions
Key architectural choices and trade-offs.

## License
MIT
```

### 11.2 Architecture Diagram (ASCII)

```text
                    ┌──────────────────────────────┐
                    │        GitHub Actions          │
                    │  CI: lint, test, smoke         │
                    │  CD: retrain on trigger        │
                    └──────────────┬───────────────┘
                                   │ triggers
                                   ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│   DVC    │───▶│  Prefect  │───▶│  MLflow   │───▶│  Model     │
│ (data)   │    │ Pipeline  │    │ (track +  │    │  Registry  │
└──────────┘    └──────────┘    │  registry) │    └─────┬─────┘
                                └──────────┘          │
                                                      │ loads champion
                                                      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│ Evidently │◀──│ Batch     │◀──│  FastAPI   │◀──│  Docker    │
│   AI     │    │ Predict   │    │  Service   │    │  Compose   │
└────┬─────┘    └──────────┘    └──────────┘    └───────────┘
     │
     ▼
┌──────────┐
│ Streamlit │
│ Dashboard │
└──────────┘
```

### 11.3 Demo Script

A `demo.sh` script runs a narrated walkthroughs:

1. Start MLflow + API + Dashboard with `docker-compose up -d`.
2. Show MLflow UI with past experiment
3. Run batch prediction on a sample batch
4. Show prediction output CSV
5. cURL the API with a sample customer
6. Open Streamlit dashboard, point out drift metrics
7. Simulate new data arrival, trigger retrain
8. Show new model version in MLflow Registry
9. Promote to Production, restart API, verify new version

### 11.4 Screen Recording Plan

3-5 minutes screen recording covering:

- Repo structure (30s)
- Pipeline execution (1 min)
- MLflow experiments + registry (30s)
- API calls (30s)
- Monitoring dashboard (30s)
- Trigger retrain + promote (30s)

Embedded in README.

### 11.5 Live Demo Option

Streamlit dashboard could be hosted on **Streamlit Community Cloud** pointing to a static snapshot of monitoring data. API could be on **Hugging Face Spaces** with a Gradio/FastAPI wrapper. Optional: not required, but nice-to-have.

---

## End of Document.

---
