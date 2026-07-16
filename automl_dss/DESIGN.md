# AUTO ML BASED SMART DECISION SUPPORT SYSTEM
## Complete Project Design Document

---

## 1. PROJECT OVERVIEW

A web-based system that accepts structured CSV datasets and automates the full ML pipeline:
upload → validate → EDA → preprocess → train multiple models → compare → select best →
predict → generate recommendations. Built with Python + Streamlit, runs locally, no cloud required.

**Target users:** Non-ML-experts, students, analysts  
**Scope:** Classification, Regression, Clustering on tabular CSV data only  
**Team size:** 2 students, 1 semester (~16 weeks)

---

## 2. SYSTEM ARCHITECTURE

```
User Browser
     │
     ▼
┌─────────────────────────────────────────────────┐
│              Streamlit Frontend (app.py)         │
│   Step-by-step wizard with sidebar progress      │
└──────────────────┬──────────────────────────────┘
                   │ session_state (in-memory)
        ┌──────────┼──────────────┐
        ▼          ▼              ▼
  ┌──────────┐ ┌──────────┐ ┌──────────────┐
  │ modules/ │ │ utils/   │ │ data/sample/ │
  │ upload   │ │ detector │ │ sample CSVs  │
  │ eda      │ └──────────┘ └──────────────┘
  │ preproc  │
  │ training │
  │ results  │
  │ predict  │
  │ recommend│
  └──────────┘
        │
        ▼
  Scikit-learn Models
  Pandas / NumPy
  Plotly Charts
```

All state is held in `st.session_state`. No database. No server-side persistence.

---

## 3. MAIN MODULES

| Module | File | Responsibility |
|---|---|---|
| App Shell | `app.py` | Step routing, sidebar progress, reset |
| Upload | `modules/upload.py` | CSV ingestion, validation, target selection |
| EDA | `modules/eda.py` | Stats, distributions, correlation, outliers |
| Preprocessing | `modules/preprocessing.py` | Missing values, encoding, scaling, feature selection |
| Training | `modules/training.py` | Multi-model training, CV, comparison table |
| Results | `modules/results.py` | Confusion matrix, ROC, residuals, cluster PCA |
| Prediction | `modules/prediction.py` | Manual input + batch CSV prediction |
| Recommendations | `modules/recommendations.py` | Rule-based decision support output |
| Detector | `utils/detector.py` | Auto problem type detection logic |

---

## 4. FOLDER STRUCTURE

```
automl_dss/
├── app.py                        # Entry point
├── requirements.txt
├── DESIGN.md                     # This document
├── modules/
│   ├── __init__.py
│   ├── upload.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── results.py
│   ├── prediction.py
│   └── recommendations.py
├── utils/
│   ├── __init__.py
│   └── detector.py
├── data/
│   └── sample/
│       ├── iris_sample.csv        # Classification test data
│       └── salary_sample.csv      # Regression test data
└── tests/
    ├── test_detector.py
    ├── test_preprocessing.py
    └── test_training.py
```

---

## 5. WORKFLOW

```
[1] Upload CSV
      │ validate (rows, cols, duplicates)
      ▼
[2] Auto-detect problem type
      │ user can override
      ▼
[3] EDA
      │ stats, distributions, correlation, outliers
      ▼
[4] Preprocessing
      │ missing value fill → encoding → scaling → feature selection
      ▼
[5] Model Training (parallel loop)
      │ train/test split → cross-validation → fit → evaluate
      ▼
[6] Model Comparison
      │ tabular comparison → bar chart → auto-select best
      ▼
[7] Results Dashboard
      │ confusion matrix / ROC / residuals / cluster PCA
      ▼
[8] Prediction
      │ manual row input or batch CSV upload
      ▼
[9] Recommendations
      │ rule-based findings + download report
```

---

## 6. PROBLEM TYPE DETECTION LOGIC

```python
def detect_problem_type(df, target_col):
    col = df[target_col].dropna()

    # Rule 1: Non-numeric dtype → Classification
    if col.dtype == object or col.dtype == bool:
        return "classification"

    n_unique = col.nunique()

    # Rule 2: Very few unique values → Classification
    if n_unique <= 10:
        return "classification"

    # Rule 3: Low cardinality ratio → Classification
    if n_unique / len(col) <= 0.05:
        return "classification"

    # Rule 4: Integer with limited unique values → Classification
    if pd.api.types.is_integer_dtype(col) and n_unique <= 30:
        return "classification"

    # Default → Regression
    return "regression"
```

User can always override the auto-detection via radio buttons.  
If no target is selected → Clustering mode.

---

## 7. MODEL SELECTION LOGIC

### Classification Models
| Model | Library |
|---|---|
| Logistic Regression | sklearn |
| Decision Tree | sklearn |
| Random Forest | sklearn |
| Gradient Boosting | sklearn |
| KNN | sklearn |
| SVM | sklearn |
| Naive Bayes | sklearn |

**Selection metric:** Highest cross-validated accuracy (StratifiedKFold)  
**Tiebreaker:** F1 weighted score

### Regression Models
| Model | Library |
|---|---|
| Linear Regression | sklearn |
| Ridge | sklearn |
| Lasso | sklearn |
| Decision Tree | sklearn |
| Random Forest | sklearn |
| Gradient Boosting | sklearn |
| KNN | sklearn |
| SVR | sklearn |

**Selection metric:** Highest cross-validated R²  
**Tiebreaker:** Lowest RMSE

### Clustering
| Model | Library |
|---|---|
| KMeans (k=2..N) | sklearn |

**Selection metric:** Highest silhouette score  
Elbow method shown for visual validation.

---

## 8. DASHBOARD FEATURES

| Step | Feature | Chart Type |
|---|---|---|
| EDA | Distributions | Histograms (Plotly) |
| EDA | Target balance | Pie chart |
| EDA | Correlation | Heatmap |
| EDA | Missing values | Bar chart |
| EDA | Outlier summary | Table |
| Results | Confusion matrix | Annotated heatmap |
| Results | ROC curve | Line chart (binary only) |
| Results | Actual vs Predicted | Scatter |
| Results | Residuals | Scatter |
| Results | Feature importance | Horizontal bar |
| Results | Model comparison | Bar chart |
| Clustering | Elbow method | Line chart |
| Clustering | Silhouette | Line chart |
| Clustering | Cluster viz | PCA 2D scatter |
| Clustering | Cluster sizes | Bar chart |
| Prediction | Class probabilities | Bar chart |

---

## 9. TECHNOLOGY STACK

| Component | Technology | Version |
|---|---|---|
| UI / App | Streamlit | ≥ 1.30 |
| Data manipulation | Pandas | ≥ 2.0 |
| Numerical ops | NumPy | ≥ 1.24 |
| ML models | Scikit-learn | ≥ 1.3 |
| Visualizations | Plotly | ≥ 5.18 |
| Language | Python | 3.10+ |
| No database | — | (session_state only) |
| No cloud | — | (localhost only) |

Install: `pip install -r requirements.txt`  
Run: `streamlit run app.py`

---

## 10. DEVELOPMENT ROADMAP

### Phase 1 — Core Pipeline (Weeks 1–4)
- [ ] Project setup, virtual environment
- [ ] Upload + validation module
- [ ] Basic EDA (stats, missing values)
- [ ] Preprocessing pipeline (missing fill, encoding)

### Phase 2 — ML Engine (Weeks 5–9)
- [ ] Classification training + CV
- [ ] Regression training + CV
- [ ] Model comparison table
- [ ] Best model selection

### Phase 3 — Visualization (Weeks 10–12)
- [ ] Confusion matrix, ROC, residual plots
- [ ] Feature importance charts
- [ ] Clustering + PCA visualization

### Phase 4 — DSS Features (Weeks 13–14)
- [ ] Prediction module (manual + batch)
- [ ] Recommendations engine
- [ ] Report download

### Phase 5 — Polish + Testing (Weeks 15–16)
- [ ] Unit tests
- [ ] Error handling and edge cases
- [ ] User testing with sample datasets
- [ ] Documentation

---

## 11. TESTING PLAN

### Unit Tests (pytest)

| Test File | What It Tests |
|---|---|
| `test_detector.py` | detect_problem_type on edge cases |
| `test_preprocessing.py` | missing fill, encoding, scaling output shapes |
| `test_training.py` | models train and return expected metric keys |

### Manual Testing Matrix

| Dataset | Task | Expected Outcome |
|---|---|---|
| iris_sample.csv | Classification | Accuracy > 90% |
| salary_sample.csv | Regression | R² > 0.80 |
| Customer data (no target) | Clustering | Silhouette > 0.30 |
| Dataset with 30% missing | Any | No crash, imputed cleanly |
| Single-class target | Classification | Warning shown |
| 5-row CSV | Any | Validation error |

### Edge Cases to Test
- CSV with all-missing column
- Target column with spaces in name
- Dataset with only 2 columns
- Categorical target with >20 classes
- All-numeric dataset for clustering

---

## 12. LIMITATIONS

These are hard constraints — not roadmap items. Be honest about them.

1. **No hyperparameter tuning** — models use default parameters only (GridSearchCV is too slow for a real-time UI without async)
2. **No deep learning** — by design; scope-appropriate
3. **Single CSV upload** — no joins, no multi-table support
4. **No model persistence** — models exist in session_state only; closing the browser loses them
5. **No authentication** — single-user local app
6. **Clustering limited to KMeans** — Agglomerative and DBSCAN present in code but only KMeans fully integrated in pipeline
7. **No time-series support** — temporal patterns are not handled
8. **Max 50,000 rows** — above this, training will be slow or memory-intensive
9. **No automated hyperparameter optimization** — would require caching layer
10. **Detection heuristic may misclassify** — user override is the safety valve

---

## 13. FUTURE ENHANCEMENTS

These are realistic next steps — not vaporware.

1. **Model export** — `joblib.dump()` to save trained model as `.pkl`
2. **Hyperparameter tuning** — Add basic GridSearchCV for top 2 models only
3. **SMOTE for class imbalance** — `imbalanced-learn` integration
4. **Feature engineering UI** — polynomial features, log transforms
5. **Report as PDF** — use `reportlab` or `pdfkit`
6. **Multi-label classification** — `MultiLabelBinarizer`
7. **Learning curves** — visualize bias/variance as training size grows
8. **SHAP explainability** — `shap` library for local feature attribution
9. **Database persistence** — SQLite to save sessions
10. **Dark/light theme toggle** — Streamlit config

---

*Document version: 1.0 | Project: AutoML DSS | Level: BS 4th Semester*
