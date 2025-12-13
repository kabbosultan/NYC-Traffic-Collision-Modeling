# Predicting Collision Severity in NYC (Vision Zero)
<img width="1156" height="644" alt="Screenshot 2025-12-12 at 1 22 37 PM" src="https://github.com/user-attachments/assets/1306c33a-af0e-44cb-870e-e757779492b0" />

## PM board
[Project Management Board](https://www.notion.so/kabbo/The-Marcy-Lab-School-DA-Fellowship-Kabbo-Ibrahima-2c3709faef5981328db5feafda5cb0dd)

## Overview
**Project Purpose:**
This project supports New York City's **Vision Zero** initiative, which aims to eliminate traffic deaths and serious injuries. By analyzing historical motor vehicle collision data, we aim to build a classification model that predicts the likelihood of a collision resulting in **Killed or Seriously Injured (KSI)** outcomes.

This predictive tool empowers Vision Zero analysts and city planners to move from **reactive** analysis (analyzing where crashes happened) to **proactive** resource allocation (identifying where severe crashes are likely to happen given specific conditions).

## Business Problem
**Stakeholder:** Vision Zero Traffic Safety Analysts & NYC Department of Transportation.

**Problem:** Traffic safety resources (enforcement, street redesigns, community outreach) are finite. Analysts need to prioritize interventions based on risk rather than just raw volume.

**Goal:** Predict whether a specific set of collision characteristics (time, borough, vehicle type, contributing factors) will result in a **KSI** event.
* **Target Definition:** `KSI = 1` (Severe) if (Fatalities > 0) OR (Injuries > 2); else `0` (Minor).
* **Why this target?** Fatalities alone are rare. Including severe injuries (2+ people injured) creates a more robust dataset (~9.01% positive class) and aligns with the goal of reducing all severe harm.

## Repository Structure
```text
nyc-mod6-project/
├── app/
│   └── streamlit_app.py                                # Streamlit Dashboard App 
├── data/                       
|   └── cleaned_motor_vehicle_collisions.csv.zip        # Cleaned dataset
├── figures/ 
|   └── 01_target_distribution.png
|   └── 02_ksi_rate_by_hour_category.png
|   └── 03_ksi_rate_by_borough.png
|   └── 04_ksi_rate_weekday_vs_weekend.png
|   └── baseline_confusion_matrices.png
|   └── simplemodel_confusion_matrices.png
|   └── tunedmodel_confusion_matrices.png
├── models/
|   └── final_model.pkl                                 # Final Model
|   └── ksi_model_meta.json                             # Meta data
├── notebooks/
│   ├── 01_eda.ipynb                                    # Exploratory Data Analysis, Cleaning & Feature Engineering
│   ├── 02_modeling_baseline.ipynb                      # Dummy Classifier & Baseline Metrics
│   ├── 03_modeling_simple.ipynb                        # Logistic Regression Model
│   ├── 04_modeling_tuned.ipynb                         # Tuned & Final Model
├── requirements.txt                                    # Python dependencies
└── README.md                                           # Project Documentation
```

## Dataset & Dictionary

  * **Source:** [NYC Open Data - Motor Vehicle Collisions - Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)
  * **Timeframe:** 2022 – Present (Filtered to ensure relevance to current traffic patterns).
  * **Volume:** \~342,418 total collisions.

| Feature Name | Description | Data Type |
| :--- | :--- | :--- |
| `crash_date` / `crash_time` | Date and time of the collision. | DateTime |
| `borough` | NYC Borough (Bronx, Brooklyn, Manhattan, Queens, Staten Island). | Categorical |
| `vehicle_type_code_1` | Type of first vehicle involved (e.g., Sedan, SUV, Bus). | Categorical |
| `contributing_factor` | Primary cause (e.g., Driver Inattention, Alcohol). | Categorical |
| `high_risk_factor` | Engineered flag for high-risk behaviors (Speeding, Alcohol, etc.). | Binary |
| `hour_category` | Time of day bucket (e.g., Morning Rush, Late Night). | Categorical |

## Reproducibility

1.  **Environment Setup:**
    ```bash
    # Clone the repo
    git clone [https://github.com/yourusername/project.git](https://github.com/yourusername/project.git)
    cd project

    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt

2.  **Data Download:**
      * Download the CSV from [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95).
      * Save it to the `data/` folder (ensure the filename matches the notebook paths, e.g., `Motor_Vehicle_Collisions_-_Crashes.csv`).
3.  **Run the Pipeline:**
      * Run `01_eda.ipynb` to clean data and generate processed files.
      * Run modeling notebooks (`02` through `04`) to train and evaluate models.
      * Run the app: `streamlit run app/streamlit_app.py`.

## Exploratory Data Analysis (Decision-Oriented)

### Stakeholder & Decisions

  * **Resource Deployment:** Where to station EMS units or traffic enforcement during high-risk windows (e.g., "Late Night" in "Brooklyn").
  * **Infrastructure Prioritization:** Identifying which boroughs or street types require traffic calming measures.

### Data Cleaning & Quality Checks

  * **Missing Data:**
      * **Boroughs:** \~7% missing. Used spatial mapping (Lat/Long → Borough) to recover critical location data.
      * **Vehicle 3+:** Columns for vehicles 3, 4, and 5 were dropped due to \>90% missingness.
      * **Contributing Factors:** Nulls imputed with "Unknown".
  * **Imbalance:** The dataset is imbalanced, with **\~9.01%** of crashes classified as Severe (KSI=1).
<img width="1699" height="1770" alt="image" src="https://github.com/user-attachments/assets/86de0db5-9215-4aac-a2bb-a0ea35f65cc4" />

### Visual Findings

  * **Geography of Risk:**
      * **Brooklyn** has the highest volume of total crashes (\~140k) and injuries.
      * **Bronx & Brooklyn** are neck-and-neck for the highest **KSI Rate**, despite differences in volume.
<img width="2070" height="1467" alt="image" src="https://github.com/user-attachments/assets/77fbdc1e-8394-421e-9218-3ed87bd320f0" />

  * **Temporal Patterns:**
      * KSI rates are lowest during morning rush hour (congestion slows traffic) and **highest at night** and on **weekends**.
    
<img width="2069" height="1467" alt="image" src="https://github.com/user-attachments/assets/8a1eed50-b791-4f5c-a703-91028527c2f4" />
<img width="2069" height="1467" alt="image" src="https://github.com/user-attachments/assets/6efc226a-f97b-4776-952c-a286ab4960b8" />

  * **Contributing Factors:**
      * "Driver Inattention/Distraction" is the leading cause (\>90k cases).
      * "Failure to Yield" is the second highest known cause.
<img width="802" height="456" alt="Screenshot 2025-12-12 at 1 05 13 PM" src="https://github.com/user-attachments/assets/bb6dc0e3-bed0-4c89-bade-c296c5fddfc7" />

### Pre-Model Assumptions

1.  **Independence:** We assume crash events are independent of each other.
2.  **Class Imbalance:** With only \~9% positive cases, accuracy is a misleading metric. We prioritize **Recall** (minimizing false negatives) and **ROC-AUC**.
3.  **Reporting Bias:** We assume "Unspecified" contributing factors are randomly distributed and do not systematically hide specific causes.

## Modeling Approach

We implemented a 3-model approach using a **Stratified Train-Test Split** to maintain class proportions.

### 1\. Baseline Model (Dummy Classifier)

  * **Role:** Statistical baseline.
  * **Strategy:** Predicts the majority class ("No KSI") for every instance.
  * **Performance:** \~91% Accuracy, but **0% Recall**.
  * **Insight:** Confirms that accuracy is not a valid metric for this problem; a "smart" model must beat this baseline by actually identifying severe cases.
<img width="1850" height="1408" alt="image" src="https://github.com/user-attachments/assets/30c12a21-2738-4d8f-a7ed-959f3dde5f79" />

### 2\. Simple Model (Logistic Regression)

  * **Role:** Interpretable baseline.
  * **Strategy:** Logistic Regression with `class_weight='balanced'`.
  * **Value:** Provides Odds Ratios, allowing stakeholders to see *how much* a factor (e.g., "SUV") increases the odds of severe injury.
  * **Goal:** Recall \> 60%.
<img width="1850" height="1408" alt="image" src="https://github.com/user-attachments/assets/445545a7-66e8-4304-9e9d-b7ef0f4729de" />

### 3\. Tuned Model (Random Forest / XGBoost)

  * **Role:** High-performance non-linear model.
  * **Strategy:** Hyperparameter tuning focusing on maximizing **Recall** and **ROC-AUC**.
  * **Value:** Captures complex interactions (e.g., *Rain* might only be dangerous at *Night* in *Brooklyn*).
  * **Goal:** Recall \> 58%, AUC \> 0.6.
<img width="1850" height="1408" alt="image" src="https://github.com/user-attachments/assets/0fe6b737-dc31-457b-810c-f5408cbb3468" />

### Evaluation Metrics Summary

| Metric | Baseline | Simple (LogReg) | Tuned Model |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 91% | 49% | 56%|
| **Recall (KSI)** | 0% | 62% | 58% |
| **Precision** | 0% | 10% | 12% |
| **ROC-AUC** | 0.50 | .56 | .60 |

## Ethics & Limitations

### Ethical Considerations

  * **Reporting Bias:** The data relies on police reports (MV-104AN). Minor crashes may be underreported in certain neighborhoods.
  * **Enforcement Bias:** Using this model to allocate *police* enforcement could reinforce existing inequities in over-policed communities. We recommend using insights primarily for **street design** and **infrastructure** improvements.
  * **Fairness:** We must ensure the model does not disproportionately flag specific neighborhoods for enforcement actions purely based on reporting volume.

### Limitations

  * **Proxy for Severity:** "Injuries \>= 2" is a heuristic for severity. It may capture minor injuries involving multiple passengers rather than one critical injury.
  * **Causality:** The model identifies correlations, not causation. A high risk in a specific borough does not inherently mean the borough causes the crash; it may be related to road design or traffic density. 

## Streamlit App

We developed a local web application to make the model accessible to non-technical stakeholders.

  * **Stakeholder Output:** The app allows an analyst to input conditions (e.g., "Brooklyn", "Raining", "Late Night") and receive a **Severity Probability Score** (0-100%) and a risk assessment.
  * # How to Use It

1. Select conditions in the left sidebar
2. Click **Predict**
3. Results:
   * No KSI probability
   * KSI probablity
4. Adjust inputs to explore different scenarios
5. Demo: https://youtu.be/v3F9HaC9-mI

## Presentation Slides
[Slides](https://docs.google.com/presentation/d/1DX8IYVHL67hMYJoV77tnIgDyl_Hys09SlTCgPZL3dpY/edit?slide=id.g3aeab37721c_0_11#slide=id.g3aeab37721c_0_11)

## Contributors
- [Ibrahima Diallo](https://www.linkedin.com/in/ibranova/): Python Developer
- [Kabbo Sultan](https://www.linkedin.com/in/kabbosultan/): Project Manager
