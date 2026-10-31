# 📅 Optimal Social Media Posting Calendar

> Predicting Facebook post engagement and generating data-driven posting calendars for Tunisia's three telecom operators: **Ooredoo**, **Orange** and **Tunisie Telecom**.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-model-EB5E28)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-charts-3F4F75?logo=plotly&logoColor=white)

Summer internship project.

---

## 🎯 Problem

When is the best time to post, and what kind of content should go out at that time?

Social media teams often choose posting times by habit or intuition. This project uses the operators' own posting history to answer that question with data. It predicts how much engagement a post will get depending on **the day of the week, the hour, and the content category**. It then turns these predictions into a **ready-to-use weekly posting calendar** for each operator, which can be explored in an interactive app.

---

## 📊 Data

About **39,000 Facebook posts** from the operators' official pages:

| Operator        | Posts  |
|-----------------|-------:|
| Orange          | 18,323 |
| Tunisie Telecom | 12,139 |
| Ooredoo         |  8,914 |

Each post includes its timestamp, message, post type, and engagement metrics. These metrics are reactions (like, love, haha, wow, sad, angry), comments, shares, views, reach (organic and paid), clicks and saves.

> ⚠️ The raw data is not included in this repository for confidentiality reasons.

---

## ⚙️ How it works

```
Raw Facebook exports ──► Data pipeline ──► XGBoost models ──► Pattern generator ──► Calendars + Streamlit app
   (3 operators)         cleaning &        global + one per    best (day, hour,      calendrier_*.csv
                         features          operator            content) slots
```

### 1. Data preparation
`notebooks/01_data_preparation.ipynb`, `src/data_pipeline.py`

- Load and harmonise the three exports, which have inconsistent schemas, types and encodings.
- Clean missing values and deleted posts.
- Build the target, **`total_engagement`**, and the features:

| Type | Features |
|------|----------|
| Timing | `hour`, `day_of_week`, `month`, `is_weekend`, `is_business_hours`, `time_slot_id` |
| Content | `content_categorie`, `message_length`, `has_hashtag`, `has_link`, `has_emoji` |
| Operator | `entreprise` (global model only) |

### 2. Modeling
`notebooks/04_modeling.ipynb`, `src/model_training.py`

- A scikit-learn **`Pipeline`** combines a `ColumnTransformer` (one-hot encoding of categorical features) with an **XGBoost regressor**.
- The target is **log-transformed**, because engagement is heavily skewed: most posts get a few interactions, and a few go viral.
- The data is split 80/20 into train and test sets.
- One **global model** and **one model per operator** are trained. They are saved with `joblib` (`models/*.pkl`), along with their metrics and features (`models/model_metadata.json`).

### 3. Model comparison
`notebooks/Test_Comparasion.ipynb`

- Compares the global and per-operator models.

### 4. Calendar generation
`src/pattern_generator.py`

- Scores every combination of day, hour and content category with the models.
- Ranks the slots by **relative** predicted engagement and keeps the best ones.
- Exports one calendar per operator (`outputs/calendrier_*.csv`).

### 5. Interactive app
`src/app.py`

- A **Streamlit** dashboard with **Plotly** charts to explore each operator's recommended posting slots and calendar.

---

## 📈 Results

Evaluated on the test set (20 %), on the log scale of `total_engagement`:

| Model           | R²    | MAE   | Test posts |
|-----------------|------:|------:|-----------:|
| Global          | 0.161 | 0.539 | 7,773 |
| Orange          | 0.189 | 0.594 | 3,568 |
| Tunisie Telecom | 0.148 | 0.446 | 2,423 |
| Ooredoo         | 0.111 | 0.532 | 1,783 |

**How to read these numbers:**

- Timing and format explain only a **modest share** of engagement. That's expected: content quality, campaigns, paid boosting and current events drive most of it, and none of them are in the features.
- The models are therefore used to **rank time slots against each other**, not to predict an exact number of likes. Comparing slots is a much easier task, and it's what the calendar needs.
- Metrics on the raw scale vary widely: R² is 0.72 for Orange but about 0 for Tunisie Telecom. A handful of viral posts dominate those scores, which is why the log scale is the main reference.


---

## 🛠️ Tools & technologies

| Category              | Tools |
|-----------------------|-------|
| Language              | Python |
| Data manipulation     | pandas, NumPy |
| Machine learning      | XGBoost, scikit-learn (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, metrics, `train_test_split`) |
| Model persistence     | joblib, JSON metadata |
| Visualisation & app   | Streamlit, Plotly |
| Exploration           | Jupyter Notebook |
| Environment & version control | VS Code, venv, Git, GitHub |

---

## 💡 What I learned

- **Real-world data is messy.** The three operators' exports didn't share the same column types or encodings: the same reaction column was text in one file and numeric in another, and the files were in cp1252 encoding. I learned to audit a dataset before trusting it and to build a cleaning pipeline that runs the same way every time.
- **Skewed targets need care.** Engagement follows a long-tailed distribution. Training on the log of engagement, and evaluating on that scale too, kept a few viral posts from dominating the model and the metrics.
- **Reading metrics honestly.** A low R² isn't automatically a failure. It depends on the question. Predicting exact engagement from timing alone is close to impossible, but ranking time slots is useful and achievable. I learned to match the evaluation to how the model will actually be used.
- **Global vs specialised models.** Training both a global model and per-operator models, then comparing them, showed me the trade-off between having more data and capturing each audience's own behaviour.
- **Building reusable ML pipelines.** Wrapping preprocessing and the model in a scikit-learn `Pipeline` means the same transformations are applied in training and prediction. Saving the models with their metadata (features, defaults, metrics) made them usable from other scripts and the app.
- **From prediction to decision.** A model score isn't useful to a marketing team on its own. Turning predictions into a calendar and an interactive Streamlit dashboard taught me to think about the end user.
- **From notebooks to a project.** I moved exploratory code from notebooks into reusable modules (`src/`), and published the project on GitHub with a proper `.gitignore`, keeping data and secrets out of the repo.

---

## 📁 Project structure

```
optimal-social-media-posting-calendar/
├── models/
│   ├── engagement_model_global.pkl
│   ├── engagement_model_Ooredoo.pkl
│   ├── engagement_model_Orange.pkl
│   ├── engagement_model_TT.pkl
│   ├── model_metadata.json
│   └── model_training_data.csv
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── Test_Comparasion.ipynb
│   └── optimal_pattern_test.csv
├── outputs/
│   ├── figures/
│   ├── calendrier_Ooredoo.csv
│   ├── calendrier_Orange.csv
│   └── calendrier_TT.csv
├── src/
│   ├── data_pipeline.py
│   ├── model_training.py
│   ├── pattern_generator.py
│   └── app.py
├── requirements.txt
└── README.md
```

---
