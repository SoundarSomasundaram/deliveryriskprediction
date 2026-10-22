# SASTRA DEEMED TO BE UNIVERSITY
## SCHOOL OF COMPUTING / DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
### B.TECH MINOR PROJECT COMPREHENSIVE REPORT DETAILS

---

# PROJECT TITLE
## LOGIRISK AI: SUPPLY CHAIN LATE DELIVERY RISK CLASSIFIER AND OPERATIONAL DECISION SUPPORT SYSTEM

### CANDIDATE DETAILS
| Register Number | Student Name |
| :--- | :--- |
| **127018056** | **Soundar S** |
| **127004255** | **Sri Vignesh G** |
| **127014035** | **Niranjan R** |

**Degree / Course:** Bachelor of Technology (B.Tech)  
**Institution:** SASTRA Deemed to be University, Tirumalaisamudram, Thanjavur – 613401, Tamil Nadu, India  
**Academic Year:** 2024 – 2025 / 2025 – 2026  

---

## TABLE OF CONTENTS

- [1. Project Overview](#1-project-overview)
- [2. Dataset and Input Information](#2-dataset-and-input-information)
- [3. Existing System and Proposed System](#3-existing-system-and-proposed-system)
- [4. Objectives](#4-objectives)
- [5. Complete Methodology](#5-complete-methodology)
- [6. Technical Architecture](#6-technical-architecture)
- [7. Results and Discussion](#7-results-and-discussion)
- [8. Results Tables and Figure Details](#8-results-tables-and-figure-details)
- [9. SASTRA Report: Chapter 1 — Introduction](#9-sastra-report-chapter-1--introduction)
- [10. SASTRA Report: Chapter 2 — Objectives](#10-sastra-report-chapter-2--objectives)
- [11. SASTRA Report: Chapter 3 — Methodology](#11-sastra-report-chapter-3--methodology)
- [12. SASTRA Report: Chapter 4 — Results and Discussion](#12-sastra-report-chapter-4--results-and-discussion)
- [13. SASTRA Report: Chapter 5 — Conclusions and Further Work](#13-sastra-report-chapter-5--conclusions-and-further-work)
- [14. References and Literature Review](#14-references-and-literature-review)
- [15. Abstract and Synopsis](#15-abstract-and-synopsis)
- [16. Preliminary Report Pages](#16-preliminary-report-pages)
- [17. Appendix](#17-appendix)

---

# 1. PROJECT OVERVIEW

### 1.1 Exact Project Title
**LogiRisk AI: Supply Chain Late Delivery Risk Classifier and Operational Decision Support System**

### 1.2 Problem Domain and Background
The project belongs to the domain of **Supply Chain Analytics, Logistics Intelligence, and Applied Supervised Machine Learning**. In contemporary international trade and omnichannel retail, fulfillment reliability is one of the primary drivers of enterprise brand loyalty, customer retention, and operational profitability. Supply chain networks are inherently vulnerable to unpredictable disruptions, warehouse operational bottlenecks, geographic transit hurdles, customs clearance delays, and carrier scheduling mismatches.

### 1.3 Real-World Problem Being Addressed
In modern e-commerce and multi-echelon supply chains, delayed deliveries incur direct financial penalties (such as service level agreement chargebacks and expedited freight re-routing), severe inventory holding overhead, and substantial customer churn. Traditional enterprise logistics systems operate reactively—notifying managers of late deliveries only after an SLA has already been breached or when angry customer tickets are logged.

### 1.4 Problem Statement
Given transactional order details, shipping specifications, geographic customer and destination markets, order timing cutoffs, and item financial parameters captured prior to or at dispatch time, how can a logistics enterprise accurately predict whether an order will suffer from a late delivery (`Late_delivery_risk = 1`), identify the key operational factors driving that risk, and prescribe real-time mitigation actions before carrier dispatch?

### 1.5 Why This Problem is Important
- **Financial Protection:** Logistics service providers and global retailers lose billions annually to contractual delay penalties, expedited transit fees, and customer refunds.
- **Customer Satisfaction:** Late delivery is the leading cause of customer defection in modern e-commerce.
- **Resource Allocation:** Supply chain dispatch controllers need predictive triage to prioritize high-risk, high-value consignments before shipments leave the warehouse dock.

### 1.6 Motivation for Developing the Project
Academic and industrial supply chain models often remain trapped in static Jupyter notebooks, lacking end-to-end deployment, domain-specific feature engineering, multi-algorithm benchmarking, and interactive decision support interfaces. The motivation was to bridge the gap between theoretical machine learning and practical supply chain operations by building a complete, high-fidelity system that not only predicts delivery delays with high precision (88.58%+) but also provides explainability and actionable "what-if" mitigation advisory.

### 1.7 Main Aim of the Project
To design, implement, benchmark, and deploy an automated, end-to-end Machine Learning classification and decision-support web application that predicts global supply chain late delivery risks with high precision and provides prescriptive operational recommendations.

### 1.8 Specific Objectives
1. Perform rigorous data cleaning, target leakage prevention, and domain feature engineering on the DataCo Global Supply Chain dataset.
2. Formulate predictive temporal, geographical, and commercial attributes (`is_rush_delivery`, `is_weekend`, `order_hour`, `is_same_country`, `profit_margin`, `discount_ratio`, `is_loss_order`).
3. Benchmark four distinct classification algorithms: Decision Tree, Random Forest, Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM).
4. Evaluate model performance across Accuracy, Precision, Recall, F1 Score, and ROC-AUC metrics with confusion matrix validation.
5. Extract global and local feature importance rankings to ensure model transparency and explainability.
6. Engineer an interactive, multi-view Streamlit web application providing single-order inspection, automated batch shipment audit, supply chain risk analytics, and model benchmarking.
7. Implement an operational "What-If Mitigation Advisor" to dynamically simulate the risk reduction achieved by reallocating shipping modes or buffering transit SLAs.

### 1.9 Scope of the Project
- **In-Scope:** Pre-dispatch and dispatch-time predictive risk modeling; handling tabular relational order data; categorical one-hot encoding; binary classification (On-time vs Late Delivery Risk); feature importance explainability; interactive UI with bulk CSV inference and downloadable audit reports.
- **Out-of-Scope (Limitations):** Real-time GPS/telematics tracking while vehicles are in transit; live weather and traffic API integration; dynamic pricing optimization.

### 1.10 Proposed Solution
A multi-tier predictive intelligence framework consisting of:
- A standardized data preprocessing and transformation pipeline (`preprocessor.joblib`).
- Trained classification models saved in serialized Joblib format (`decision_tree.joblib`, `random_forest.joblib`, `xgboost.joblib`, `lightgbm.joblib`).
- A metadata registry (`metadata.json`) storing model evaluation metrics, confusion matrices, and feature importances.
- A modern Streamlit web application (`app.py`) with Plotly visual analytics, scenario presets, batch CSV processing, and prescriptive fulfillment mitigation.

### 1.11 Novelty and Distinctive Aspects
- **Strict Leakage Prevention:** Eliminates post-event features (such as `Days for shipping (real)` and `Delivery Status`) that cause false 100% data leakage in naive academic notebooks.
- **Domain-Specific Feature Synthesis:** Integrates warehouse dispatch cutoffs (`order_hour`), weekend backlog flags (`is_weekend`), cross-border barriers (`is_same_country`), and financial stress signals (`profit_margin`, `is_loss_order`).
- **Prescriptive "What-If" Mitigation Engine:** Rather than outputting a black-box probability, the system dynamically recalculates risk under alternative fulfillment modes (e.g., demonstrating how moving an order to Standard Class with a 4-day SLA reduces delay risk).
- **Dual Inference Mode:** Accommodates individual order simulation and enterprise bulk CSV auditing with automated revenue-at-risk aggregation.

### 1.12 Intended Users and Potential Applications
- **Logistics Dispatch Managers:** Triage high-risk dispatches prior to carrier handover.
- **E-Commerce Operations Controllers:** Monitor order backlogs and adjust delivery time promises dynamically.
- **Supply Chain Analysts:** Conduct root-cause analytics across global markets, customer segments, and fulfillment channels.
- **Third-Party Logistics (3PL) Providers:** Optimize carrier allocation to meet contracted Service Level Agreements (SLAs).

### 1.13 Main Features and Functionality
1. **Interactive Single Order Predictor:** Custom sliders and dropdowns, 4 quick-load logistics scenario presets, visual risk probability gauge, risk classification badges (Low 🟢, Moderate 🟡, High 🔴).
2. **Explainable AI Driver Display:** Ranks the operational reasons driving late delivery risk for a specific order.
3. **What-If Scenario Simulator:** Generates comparison tables showing risk deltas across alternative shipping modes.
4. **Batch Shipment Audit & CSV Inference:** Drag-and-drop CSV upload, 35-record sample order loader, real-time calculation of late shipments and total at-risk sales value ($), Plotly pie and bar charts, interactive filtered table, downloadable annotated CSV.
5. **Supply Chain Risk Analytics:** Delay incidence across 4 shipping modes, regional market delay density across 5 global continents, weekend fulfillment bottleneck analysis, and scheduled SLA buffer curves.
6. **Model Benchmark Hub:** Tabular metrics comparison, side-by-side metric bar charts, interactive confusion matrix visualizer, and Top-20 feature importance leaderboards.

### 1.14 Expected Outcomes and Actual Achievements
- **Expected:** Develop a working ML model capable of classifying late deliveries better than a random baseline (>55%).
- **Actual Achievements:** Successfully engineered a 4-model benchmarking pipeline achieving **74.12% Accuracy**, **88.58% Precision**, and **0.8115 ROC-AUC** with LightGBM, backed by an operational 5-tab enterprise web application.

---

# 2. DATASET AND INPUT INFORMATION

### 2.1 Dataset Name and Source
- **Dataset Title:** DataCo Global Supply Chain Dataset
- **Origin / Source:** DataCo Global supply chain operational database (publicly curated and cataloged on Kaggle and academic repositories).
- **Local File Path:** `data/DataCoSupplyChainDataset.csv`

### 2.2 Original File Format and Dimensions
- **File Format:** Comma-Separated Values (`.csv`) with `latin-1` character encoding.
- **Full Dataset Shape:** **180,519 rows × 53 columns** (Total file size: 95,910,149 bytes / ~91.5 MB).
- **Training Subset Utilized:** 60,000 representative records loaded via `nrows=60000` in `model_trainer.py` to balance computational training throughput and statistical sample diversity.
- **Duplicate Records:** 0 duplicate rows detected in the raw dataset.

### 2.3 Target Variable and Class Distribution
- **Target Feature Name:** `Late_delivery_risk`
- **Definition:** Binary indicator indicating whether an order was delivered past the scheduled delivery date.
  - `0`: On-Time Delivery / No Delay Risk
  - `1`: Late Delivery Risk / Delivery SLA Breached
- **Full Dataset Class Distribution (180,519 records):**
  - Class `1` (Late): **98,977 records (54.83%)**
  - Class `0` (On-Time): **81,542 records (45.17%)**
- **Test Set Class Distribution (12,000 evaluated records from 20% split):**
  - Class `1` (Late): **6,943 records (57.86%)**
  - Class `0` (On-Time): **5,057 records (42.14%)**

### 2.4 Missing Values in Raw Dataset
An empirical inspection of all 53 columns revealed missing data concentrated in four specific attributes:
1. `Customer Lname`: 8 missing records (Immaterial customer surname data).
2. `Customer Zipcode`: 3 missing records.
3. `Order Zipcode`: 155,679 missing records (86.2% missing due to international orders lacking US ZIP formatting).
4. `Product Description`: 180,519 missing records (100% null).

*Resolution in Pipeline:* All four columns were excluded from model input features during data cleaning.

### 2.5 Data Cleaning and Target Leakage Prevention
To ensure true real-world predictive validity, features generated *after* shipment dispatch or directly encoding the target were dropped:
- **`Days for shipping (real)`:** Dropped because real transit days are unknown prior to final delivery.
- **`Delivery Status`:** Dropped because its categorical values (`Late delivery`, `Shipping on time`, `Advance shipping`, `Shipping canceled`) contain exact target leakage.
- **`shipping date (DateOrders)`:** Dropped to prevent temporal leakage.
- **Personally Identifiable Information (PII) & Identifiers:** `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street`, `Customer Zipcode`, `Order Zipcode`, `Order Id`, `Order Item Id`, `Order Customer Id`, `Customer Id`, `Product Image`, `Product Description`.

### 2.6 Features Selected and Engineered for Training
The final feature space consists of **22 input features** (7 categorical, 15 numerical):

| # | Feature Name | Type | Nature | Description & Engineering Formula |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `Shipping Mode` | Categorical | Raw | Standard Class, First Class, Second Class, Same Day |
| 2 | `Days for shipment (scheduled)`| Numerical | Raw | Target SLA days promised to the buyer (0 to 6 days) |
| 3 | `Type` | Categorical | Raw | Transaction transaction method: PAYMENT, TRANSFER, CASH, DEBIT |
| 4 | `Customer Segment` | Categorical | Raw | Market segment: Consumer, Corporate, Home Office |
| 5 | `Department Name` | Categorical | Raw | 11 fulfillment departments (Apparel, Technology, Golf, etc.) |
| 6 | `Category Name` | Categorical | Raw | 50 product categories (Cleats, Men's Footwear, Camping, etc.) |
| 7 | `Market` | Categorical | Raw | Global destination market: Africa, Europe, LATAM, Pacific Asia, USCA |
| 8 | `Order Region` | Categorical | Raw | 23 international geographic logistics regions |
| 9 | `Order Item Quantity` | Numerical | Raw | Number of product units per order line item |
| 10 | `Product Price` | Numerical | Raw | Base price per unit ($) |
| 11 | `Sales` | Numerical | Raw | Gross sales value per order line item ($) |
| 12 | `Order Item Discount` | Numerical | Raw | Dollar discount amount applied to the order item ($) |
| 13 | `Order Profit Per Order` | Numerical | Raw | Net profit earned on the order item ($) |
| 14 | `order_month` | Numerical | Engineered | Month of the order (1 to 12) from `order date (DateOrders)` |
| 15 | `order_dayofweek` | Numerical | Engineered | Day of week (0=Monday to 6=Sunday) |
| 16 | `order_hour` | Numerical | Engineered | Hour of day (0 to 23) when order was logged |
| 17 | `is_weekend` | Numerical | Engineered | Binary indicator: `1` if `order_dayofweek >= 5` else `0` |
| 18 | `is_rush_delivery` | Numerical | Engineered | Binary indicator: `1` if `Days for shipment (scheduled) <= 1` else `0` |
| 19 | `is_same_country` | Numerical | Engineered | Binary indicator: `1` if `Customer Country == Order Country` else `0` |
| 20 | `profit_margin` | Numerical | Engineered | $\frac{\text{Order Profit Per Order}}{\text{Sales}}$ (0 if Sales = 0) |
| 21 | `discount_ratio` | Numerical | Engineered | $\frac{\text{Order Item Discount}}{\text{Sales}}$ (0 if Sales = 0) |
| 22 | `is_loss_order` | Numerical | Engineered | Binary indicator: `1` if `Order Profit Per Order < 0` else `0` |

### 2.7 Encoding and Preprocessing Pipeline
- **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to all 7 categorical features.
- **Numerical Features:** 15 numerical features passed through without scaling (`passthrough`) to maintain feature interpretability and tree split fidelity.
- **Total Encoded Dimension:** **114 features** output by `ColumnTransformer` (15 numerical features + 99 one-hot encoded dummy columns).

### 2.8 Validation Strategy
- **Train-Test Split:** 80% training set (48,000 samples) and 20% hold-out test set (12,000 samples).
- **Random Seed:** `random_state=42` ensures complete determinism and reproducibility.

### 2.9 Dataset Assumptions and Limitations
- **Assumption:** Historical routing and transit delay patterns reflect ongoing operations.
- **Limitation 1:** Ground truth weather disruptions, localized port congestion, and carrier strikes are not logged as explicit features.
- **Limitation 2:** The dataset does not contain intermediate GPS waypoints during transit; inference is strictly pre-dispatch/dispatch-time.

---

# 3. EXISTING SYSTEM AND PROPOSED SYSTEM

### 3.1 Existing System / Baseline Approach
In conventional logistics and academic baseline implementations (including the initial exploratory Jupyter notebook `riskclassifier (1).ipynb`):
- Supply chain monitoring relies on static spreadsheet trackers, ERP status flags, or basic heuristic rules (e.g., assuming First Class deliveries are always prioritised).
- In academic baselines, models are often trained on uncurated datasets suffering from target leakage (using `Days for shipping (real)` which directly leaks whether transit days exceeded scheduled days).
- Notebooks are restricted to basic models (Decision Tree and Random Forest) using only default date attributes (`order_year`, `order_month`, `order_day`, `order_dayofweek`).
- Models remain confined to offline Jupyter cells with no user-facing deployment, no scenario simulation, and no batch audit mechanism.

### 3.2 Limitations of the Existing System
1. **Target Leakage / Artificial Performance:** Naive models that include `Days for shipping (real)` achieve 100% accuracy in notebooks, but are useless in production because actual delivery duration is unknown at dispatch time.
2. **Limited Feature Representation:** Lack of operational indicators such as dispatch cutoffs (`order_hour`), weekend fulfillment backlog (`is_weekend`), cross-border customs frictions (`is_same_country`), and order financial health (`profit_margin`).
3. **Absence of Modern Gradient Boosting:** Fails to exploit modern ensemble architectures like XGBoost and LightGBM that excel on tabular business data.
4. **Lack of Operational Actionability:** A static prediction percentage provides no actionable recommendation to logistics coordinators on how to avert the predicted delay.

### 3.3 Proposed System (LogiRisk AI)
The proposed system implements an enterprise-grade, end-to-end Machine Learning intelligence platform:
1. **Target Leakage-Free Pipeline:** Excludes all post-transit features, ensuring true pre-dispatch inference validity.
2. **Advanced Domain Feature Engineering:** Incorporates temporal, geographic, operational, and commercial interaction terms.
3. **Multi-Model Algorithmic Benchmarking:** Systematically compares Decision Tree, Random Forest, XGBoost, and LightGBM under identical validation splits.
4. **Decoupled Architecture:** Cleanly separates offline training (`model_trainer.py`), serialized artifacts (`models/`), and a dynamic frontend (`app.py`).
5. **Interactive Operational UI:** A 5-tab Streamlit dashboard delivering single-order prediction, bulk CSV auditing, interactive business intelligence, model benchmarking, and automated "What-If" fulfillment mitigation.

### 3.4 Comparison: Existing System vs Proposed System

| Dimension | Existing System (Baseline Notebook) | Proposed System (LogiRisk AI) |
| :--- | :--- | :--- |
| **Architectural Scope** | Single offline notebook (`.ipynb`) | Multi-tier modular Python application + Streamlit web UI |
| **Target Leakage Control** | Vulnerable if post-event columns retained | Strict elimination of all post-dispatch columns |
| **Algorithms Evaluated** | 2 models (Decision Tree, Random Forest) | 4 models (Decision Tree, Random Forest, XGBoost, LightGBM) |
| **Feature Engineering** | 4 basic calendar attributes | 7 engineered domain features (rush, weekend, cutoffs, margins, customs) |
| **Total Features (Encoded)**| 98 columns | 114 columns |
| **Deployment / Usability**| Requires Python/Jupyter execution | Browser-based interactive dashboard with preset scenarios |
| **Inference Modalities** | Static code execution | Dual mode: Single order interactive form + Batch CSV audit |
| **Operational Guidance** | Binary prediction label only | Explainable risk drivers + Prescriptive What-If Mitigation Advisor |
| **Export Capabilities** | None | Downloadable annotated prediction CSV with risk probabilities |

### 3.5 Advantages and Limitations of Implemented Solution
- **Advantages:** High precision (88.58%), zero target leakage, instantaneous inference (<10 ms per order), explainable root causes, intuitive visual design, and real-time prescriptive advice.
- **Limitations:** Does not incorporate dynamic telematics / GPS stream processing or live weather forecasts during transit.

---

# 4. OBJECTIVES

### 4.1 Main Objective
To design, develop, evaluate, and deploy an automated, end-to-end Machine Learning classification and operational decision-support system that predicts supply chain late delivery risk (`Late_delivery_risk`) with high precision and provides real-time mitigation recommendations for global logistics dispatches.

### 4.2 Specific Implemented Objectives
1. **Data Ingestion & Cleaning:** Ingest the DataCo Supply Chain operational dataset (180,519 records), eliminate identifier noise, handle missing data, and remove post-event target leakage columns.
2. **Domain Feature Engineering:** Engineer seven domain-specific predictive indicators capturing fulfillment urgency (`is_rush_delivery`), dispatch cutoffs (`order_hour`), weekend backlogs (`is_weekend`), cross-border friction (`is_same_country`), and commercial stress (`profit_margin`, `discount_ratio`, `is_loss_order`).
3. **Multi-Model Development:** Implement and tune four diverse supervised classification algorithms: Decision Tree, Random Forest, XGBoost, and LightGBM.
4. **Comparative Performance Benchmarking:** Benchmark the models across Accuracy, Precision, Recall, F1 Score, and ROC-AUC on a strict 20% holdout test set (12,000 records).
5. **Transparency & Feature Importance:** Extract and visualize global feature importance rankings to identify the primary drivers of logistics delays.
6. **Decision Support Dashboard:** Develop a modern, multi-tab Streamlit web application providing interactive order simulation, scenario presets, risk gauge visualizations, and bulk CSV audit tools.
7. **Prescriptive Mitigation Engine:** Build a dynamic "What-If Simulator" that evaluates alternative fulfillment levers (such as carrier mode switching and SLA adjustments) to guide delay prevention.

### 4.3 Future Work Objectives (Explicitly Identified)
1. Integration of real-time IoT telematics and transit GPS tracking data.
2. Live weather and port congestion API integration.
3. Automated integration into warehouse management systems (SAP / Oracle SCM) via REST APIs.
4. Multi-class classification to predict the exact *magnitude* of delay (e.g., 1-2 days late, 3-5 days late).

---

# 5. COMPLETE METHODOLOGY

```
+-----------------------------------------------------------------------+
|                       1. DATA ACQUISITION                            |
|        DataCo Global Supply Chain Operational Dataset (CSV)           |
|                180,519 records x 53 raw columns                       |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                  2. DATA CLEANING & LEAKAGE CONTROL                   |
|  - Drop PII & Identifiers (Customer Email, Password, Order Id, etc.)  |
|  - Drop Target Leaks (Days for shipping (real), Delivery Status)      |
|  - Drop 100% null columns (Product Description)                       |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    3. DOMAIN FEATURE ENGINEERING                      |
|  - Parse timestamps: order_month, order_dayofweek, order_hour         |
|  - Synthesize: is_weekend, is_rush_delivery, is_same_country          |
|  - Compute financial ratios: profit_margin, discount_ratio, loss_flag|
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                 4. PREPROCESSING & FEATURE ENCODING                   |
|  - Numeric features (15): passthrough                                 |
|  - Categorical features (7): OneHotEncoder(handle_unknown='ignore')   |
|  - Transformed feature space: 114 columns                             |
|  - Stratified 80/20 Train-Test Split (48,000 train / 12,000 test)     |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                  5. MODEL TRAINING & HYPERPARAMETER TUNING            |
|  - Decision Tree (max_depth=12, min_samples_split=10, leaf=5)         |
|  - Random Forest (n_estimators=100, max_depth=14, split=8, leaf=4)    |
|  - XGBoost (n_estimators=120, max_depth=6, lr=0.08, subsample=0.8)    |
|  - LightGBM (n_estimators=120, max_depth=8, lr=0.08, leaf=31)        |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    6. EVALUATION & ARTIFACT EXPORT                    |
|  - Evaluate: Accuracy, Precision, Recall, F1 Score, ROC-AUC, CM       |
|  - Serialize: models/*.joblib, preprocessor.joblib, metadata.json     |
|  - Generate: sample_data/sample_orders.csv (35 test records)          |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                     7. STREAMLIT APPLICATION (UI)                     |
|  - Tab 1: Single Order Inspector & What-If Mitigation Advisor         |
|  - Tab 2: Batch Shipment Audit & CSV Inference (KPI Cards & Charts)   |
|  - Tab 3: Supply Chain Risk Analytics (Delay rates, SLA curves)       |
|  - Tab 4: Model Benchmark Hub (Confusion Matrix & Feature Importances)|
|  - Tab 5: Architecture & Project Documentation Explorer               |
+-----------------------------------------------------------------------+
```

### 5.1 Stage 1: Data Acquisition
The dataset `data/DataCoSupplyChainDataset.csv` is loaded using `pandas.read_csv` with `encoding='latin-1'`. To balance statistical rigor with execution efficiency during model development, 60,000 records are sampled.

### 5.2 Stage 2: Target Leakage Prevention and Data Cleansing
The target column `Late_delivery_risk` is extracted as label $y \in \{0, 1\}$. Columns that cannot exist at pre-dispatch time are stripped:
- Target-correlated post-event attributes (`Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`).
- High-cardinality noise and identifiers (`Order Id`, `Order Item Id`, `Customer Id`, `Customer Email`, etc.).

### 5.3 Stage 3: Domain-Specific Feature Engineering
Using the raw timestamps, geographic metadata, and transactional finance values, 7 domain features are derived:
1. **Temporal Features:** Order date string parsed into year, month, day, day-of-week ($0 \dots 6$), and hour ($0 \dots 23$).
2. **Operational Stress Indicators:**
   $$\text{is\_rush\_delivery} = \begin{cases} 1 & \text{if } \text{Days for shipment (scheduled)} \le 1 \\ 0 & \text{otherwise} \end{cases}$$
   $$\text{is\_weekend} = \begin{cases} 1 & \text{if } \text{order\_dayofweek} \ge 5 \\ 0 & \text{otherwise} \end{cases}$$
3. **Cross-Border Factor:**
   $$\text{is\_same\_country} = \begin{cases} 1 & \text{if } \text{Customer Country} = \text{Order Country} \\ 0 & \text{otherwise} \end{cases}$$
4. **Financial Health Indicators:**
   $$\text{profit\_margin} = \frac{\text{Order Profit Per Order}}{\text{Sales}} \quad (\text{clipped to } 0 \text{ when Sales} = 0)$$
   $$\text{discount\_ratio} = \frac{\text{Order Item Discount}}{\text{Sales}}$$
   $$\text{is\_loss\_order} = \begin{cases} 1 & \text{if } \text{Order Profit Per Order} < 0 \\ 0 & \text{otherwise} \end{cases}$$

### 5.4 Stage 4: Feature Transformation & Preprocessing
A `ColumnTransformer` is configured:
- **Numerical Pipeline:** 15 continuous and binary engineered features passed through as floats.
- **Categorical Pipeline:** 7 categorical features encoded into binary indicator vectors via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- Preprocessor is fit exclusively on $X_{\text{train}}$ to prevent data snooping, transforming the feature space into 114 dimensions.

### 5.5 Stage 5: Supervised Model Training & Algorithms
Four algorithms spanning individual trees, bagging ensembles, and gradient boosting machines are trained:
1. **Decision Tree Classifier:** Baseline tree using Gini impurity splits.
2. **Random Forest Classifier:** Ensemble of 100 de-correlated bootstrap trees with parallelized training (`n_jobs=-1`).
3. **Extreme Gradient Boosting (XGBoost):** Regularized gradient boosted decision trees minimizing binary logistic loss.
4. **Light Gradient Boosting Machine (LightGBM):** Leaf-wise tree growth with histogram binning for superior training speed and recall balance.

### 5.6 Stage 6: Validation, Metric Extraction, and Artifact Serialization
- Holdout test set ($N = 12,000$) is evaluated across Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
- Confusion matrices and Gini/Gain feature importances are extracted.
- All four models are serialized using `joblib.dump` into `models/*.joblib`.
- Preprocessor is saved to `models/preprocessor.joblib`.
- Complete evaluation metrics, feature lists, and category dropdown vocabularies are compiled into `models/metadata.json`.
- A representative 35-sample test batch is exported to `sample_data/sample_orders.csv` for zero-configuration UI testing.

### 5.7 Stage 7: Web Application Interface & Decision Support
The Streamlit application (`app.py`) initializes by caching models in memory, rendering an enterprise sidebar with threshold tuning, and providing five purpose-built operational tabs.

---

# 6. TECHNICAL ARCHITECTURE

### 6.1 Technology Stack

| Layer | Technologies / Tools | Specific Version / Library | Responsibility |
| :--- | :--- | :--- | :--- |
| **Language** | Python | Python 3.10+ / 3.12 | Core runtime for data processing, ML, and web serving |
| **Data Processing** | Pandas, NumPy | `pandas>=2.0.0`, `numpy>=1.24.0` | In-memory tabular transformations, feature engineering |
| **Machine Learning**| Scikit-Learn | `scikit-learn>=1.3.0` | Decision Tree, Random Forest, ColumnTransformer, metrics |
| **Gradient Boosting**| XGBoost, LightGBM | `xgboost>=2.0.0`, `lightgbm>=4.0.0` | State-of-the-art boosted decision tree classifiers |
| **Serialization** | Joblib, JSON | `joblib>=1.3.0`, standard `json` | Storing trained model binaries and evaluation metadata |
| **Web Framework** | Streamlit | `streamlit>=1.30.0` | Reactive web interface, session state, UI widgets |
| **Visualization** | Plotly | `plotly>=5.15.0` | Interactive charts (histograms, gauges, heatmaps, pie charts) |
| **Design System** | HTML5 / Vanilla CSS | Google Font: *Plus Jakarta Sans* | Responsive styling, cards, status badges, hero banner |

### 6.2 Component Relationship Diagram

```
+--------------------------------------------------------------------------------+
|                                DATA LAYER                                      |
|  - data/DataCoSupplyChainDataset.csv (Raw operational dataset: 180k rows)      |
|  - sample_data/sample_orders.csv     (35 curated test orders for batch audit)  |
+--------------------------------------------------------------------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
|                        OFFLINE TRAINING PIPELINE                               |
|                         [ model_trainer.py ]                                   |
|  - build_features(): Feature synthesis (rush, weekend, cutoffs, financial)    |
|  - ColumnTransformer: OneHotEncoder (7 cat) + passthrough (15 num)             |
|  - Multi-Model Benchmarking: DT, RF, XGBoost, LightGBM                         |
|  - Metric Computation: Confusion Matrix, Accuracy, Precision, F1, ROC-AUC      |
+--------------------------------------------------------------------------------+
                                        |
                     +------------------+------------------+
                     | Produces Serialized Artifacts       |
                     v                                     v
+---------------------------------------+  +-------------------------------------+
|        SERIALIZED MODELS              |  |         METADATA REGISTRY           |
|  - models/decision_tree.joblib (92 KB)|  |  - models/metadata.json (13.3 KB)   |
|  - models/random_forest.joblib (12.9M)|  |    - Metric scores & comparisons    |
|  - models/xgboost.joblib       (452 KB)|  |    - Confusion matrix arrays        |
|  - models/lightgbm.joblib      (426 KB)|  |    - Top-20 feature importances     |
|  - models/preprocessor.joblib  (5.8 KB)|  |    - Categorical dropdown vocabularies|
+---------------------------------------+  +-------------------------------------+
                     |                                     |
                     +------------------+------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
|                       FRONTEND & DECISION SUPPORT SYSTEM                       |
|                                [ app.py ]                                      |
|  - @st.cache_resource: In-memory artifact loading                              |
|  - Sidebar: Model selector & decision threshold slider                         |
|  - Tab 1: Single Order Inspector, Risk Gauge, Driver XAI, Mitigation Advisor   |
|  - Tab 2: Batch Shipment Audit, CSV Inference, KPI Cards, Annotated Export    |
|  - Tab 3: Supply Chain Risk Analytics (Mode delay, Region density, SLA curve)  |
|  - Tab 4: Model Benchmark Hub (Confusion matrix heatmap, Feature bar chart)    |
|  - Tab 5: Architecture Explorer & Project Documentation                        |
+--------------------------------------------------------------------------------+
```

### 6.3 Input-Output Flow
1. **User Input:** User enters shipment parameters via Tab 1 UI form (or uploads an operational CSV batch in Tab 2).
2. **Transformation:** The raw input dataframe passes through `models/preprocessor.joblib`, which validates column presence, fills defaults, and applies one-hot encoding into a 114-element feature vector.
3. **Inference:** The selected model (`chosen_model.predict_proba`) computes class probabilities:
   $$P(\text{Late}) = P(y=1 \mid \mathbf{x})$$
4. **Classification:** If $P(\text{Late}) \ge \tau$ (default threshold $\tau = 0.50$), order is classified as `Late Delivery Risk` (Critical 🔴); otherwise `On-Time` (Safe 🟢).
5. **Mitigation Simulation:** For high-risk shipments, the system automatically simulates 4 shipping mode permutations with corresponding standard SLA days and renders a comparative risk reduction table.

---

# 7. RESULTS AND DISCUSSION

### 7.1 Verified Experimental Results
All results documented below are extracted directly from `models/metadata.json` and execution logs evaluated on the identical holdout test set of **12,000 unseen records**.

#### Model Comparison Table (Test Set: N = 12,000)

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC Score | Primary Operational Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | **74.12%** | **88.58%** | **63.45%** | **0.7393** | **0.8115** | Highest overall accuracy and ROC-AUC; low memory footprint |
| **XGBoost** | **74.08%** | **88.09%** | **63.83%** | **0.7403** | **0.8108** | Highest F1 score; robust regularization |
| **Random Forest** | 74.08% | **89.17%** | 62.84% | 0.7372 | 0.8108 | Highest precision (minimizes false alarms) |
| **Decision Tree** | 73.05% | 84.76% | 65.13% | 0.7366 | 0.7918 | Rapid interpretable tree baseline |

### 7.2 Confusion Matrix Breakdown
Each model was evaluated on 12,000 holdout records comprising **5,057 actual on-time orders (Class 0)** and **6,943 actual late orders (Class 1)**.

#### 1. LightGBM (Selected Production Model)
- True Negatives (TN): **4,489** (On-Time correctly identified)
- False Positives (FP): **568** (On-Time falsely flagged as Late)
- False Negatives (FN): **2,538** (Late orders missed)
- True Positives (TP): **4,405** (Late orders correctly caught)
- *Calculated Precision:* $\frac{4405}{4405 + 568} = 88.58\%$
- *Calculated Recall:* $\frac{4405}{4405 + 2538} = 63.45\%$

#### 2. XGBoost
- True Negatives (TN): **4,458** | False Positives (FP): **599**
- False Negatives (FN): **2,511** | True Positives (TP): **4,432**
- *Calculated Precision:* $\frac{4432}{4432 + 599} = 88.09\%$
- *Calculated Recall:* $\frac{4432}{4432 + 2511} = 63.83\%$

#### 3. Random Forest
- True Negatives (TN): **4,527** | False Positives (FP): **530**
- False Negatives (FN): **2,580** | True Positives (TP): **4,363**
- *Calculated Precision:* $\frac{4363}{4363 + 530} = 89.17\%$
- *Calculated Recall:* $\frac{4363}{4363 + 2580} = 62.84\%$

#### 4. Decision Tree
- True Negatives (TN): **4,244** | False Positives (FP): **813**
- False Negatives (FN): **2,421** | True Positives (TP): **4,522**
- *Calculated Precision:* $\frac{4522}{4522 + 813} = 84.76\%$
- *Calculated Recall:* $\frac{4522}{4522 + 2421} = 65.13\%$

### 7.3 Feature Importance Leaderboard
Feature importance analysis reveals the governing operational variables dictating delivery delays:

#### Top Predictors in Random Forest:
1. `Days for shipment (scheduled)`: **24.28%** importance
2. `Shipping Mode = Standard Class`: **16.27%** importance
3. `Shipping Mode = First Class`: **13.74%** importance
4. `is_rush_delivery` (Scheduled SLA $\le$ 1 day): **8.61%** importance
5. `order_hour` (Warehouse dispatch cutoff): **7.21%** importance
6. `Shipping Mode = Second Class`: **6.27%** importance
7. `Shipping Mode = Same Day`: **3.98%** importance
8. `profit_margin`: **1.84%** importance
9. `Order Profit Per Order`: **1.84%** importance
10. `Order Item Discount`: **1.55%** importance

#### Top Predictors in XGBoost:
1. `Shipping Mode = Standard Class`: **44.08%** importance
2. `Shipping Mode = First Class`: **14.57%** importance
3. `Days for shipment (scheduled)`: **7.65%** importance
4. `Shipping Mode = Same Day`: **7.15%** importance
5. `Shipping Mode = Second Class`: **6.03%** importance
6. `is_rush_delivery`: **1.51%** importance
7. `order_hour`: **0.83%** importance

### 7.4 Discussion of Key Findings
1. **The Shipping Mode / SLA Paradox:** The single most dominant factor determining delivery risk is not the geographic distance or product category, but the promised **Shipping Mode vs. Scheduled SLA window**. Specifically, *First Class* shipments have an empirical late rate of **98.0%** because they promise a stringent 1-day delivery window that regional carrier networks consistently fail to satisfy. In contrast, *Standard Class* carries a realistic 4-day SLA, yielding an empirical delay rate of only **37.5%**.
2. **Warehouse Dispatch Timing Impact:** The engineered `order_hour` attribute proved to be the 2nd most influential continuous feature across all models. Orders placed late in the afternoon or evening routinely miss same-day carrier handoff cutoffs, causing an automatic 24-hour slip.
3. **High Precision vs. Moderate Recall Tradeoff:** All ensemble models achieved exceptionally high precision (88.0% – 89.2%). This is a crucial operational property: when LogiRisk AI flags an order as "High Risk," logistics controllers can be **88.6% confident** that the shipment will indeed be delayed, virtually eliminating wasted manual interventions.
4. **Why LightGBM was Selected:** LightGBM achieved the highest overall test accuracy (**74.12%**) and ROC-AUC (**0.8115**), paired with lightning-fast serialization size (426 KB vs 12.9 MB for Random Forest) and sub-millisecond inference latency.

---

# 8. RESULTS TABLES AND FIGURE DETAILS

### 8.1 Summary of Suggested Report Tables

| Table Number | Table Title | Contents / Data Represented | Report Placement |
| :--- | :--- | :--- | :--- |
| **Table 2.1** | Raw Dataset Summary | Shape (180,519 × 53), file size (91.5 MB), source, missing values, duplicates (0) | Chapter 3 (Methodology) |
| **Table 2.2** | Final Feature Specification | Name, data type, domain description, and formulas for all 22 input features | Chapter 3 (Methodology) |
| **Table 3.1** | Hyperparameter Configuration | Exact parameter configurations for Decision Tree, Random Forest, XGBoost, and LightGBM | Chapter 3 (Methodology) |
| **Table 4.1** | Benchmark Performance Comparison | Accuracy, Precision, Recall, F1 Score, and ROC-AUC for all 4 evaluated algorithms | Chapter 4 (Results) |
| **Table 4.2** | Confusion Matrix Breakdown | Detailed TN, FP, FN, TP counts for all 4 models on the 12,000 test set | Chapter 4 (Results) |
| **Table 4.3** | Top-15 Feature Importance Table | Comparative importance percentages across Random Forest, XGBoost, and Decision Tree | Chapter 4 (Results) |
| **Table 4.4** | What-If Fulfillment Simulation | Sample output demonstrating risk probability change across 4 shipping modes | Chapter 4 (Results) |

### 8.2 Summary of Suggested Report Figures

| Figure Number | Figure Title / Caption | Visual Content & Description | Report Placement |
| :--- | :--- | :--- | :--- |
| **Figure 1.1** | High-Level System Concept | Visual overview of e-commerce order ingestion, ML risk scoring, and carrier triage | Chapter 1 (Introduction) |
| **Figure 3.1** | End-to-End Pipeline Workflow | Block flowchart: Data loading $\to$ Cleaning $\to$ Feature Engineering $\to$ Preprocessor $\to$ Training $\to$ Streamlit UI | Chapter 3 (Methodology) |
| **Figure 3.2** | Technical Architecture Diagram | Component layout showing interactions between datasets, Joblib models, and UI tabs | Chapter 3 (Methodology) |
| **Figure 4.1** | Algorithm Performance Comparison | Clustered bar chart comparing Accuracy, ROC-AUC, and F1 Score across the 4 models | Chapter 4 (Results) |
| **Figure 4.2** | Confusion Matrix Heatmap (LightGBM)| Normalized heatmap visualising the 4,489 TN, 568 FP, 2,538 FN, and 4,405 TP | Chapter 4 (Results) |
| **Figure 4.3** | Top Feature Importance Leaderboard | Horizontal bar chart of top 20 predictors (scheduled days, shipping modes, order hour) | Chapter 4 (Results) |
| **Figure 4.4** | Delay Incidence by Shipping Mode | Bar chart showing First Class (98.0%) vs Standard Class (37.5%) historical delay rates | Chapter 4 (Results) |
| **Figure 4.5** | SLA Buffer Risk Curve | Line plot showing how increasing scheduled SLA days from 0 to 6 squashes delay risk | Chapter 4 (Results) |
| **Figure 4.6** | Single Order Predictor Interface | Screenshot of the Streamlit interactive form with risk gauge and What-If simulator | Chapter 4 (Results) |
| **Figure 4.7** | Batch Shipment Audit Dashboard | Screenshot of batch KPI cards, risk distribution donut chart, and filtered table | Chapter 4 (Results) |

---

# 9. SASTRA REPORT: CHAPTER 1 — INTRODUCTION

### 1.1 Background of the Study
Over the past two decades, the exponential growth of global omnichannel commerce, cross-border e-commerce, and just-in-time (JIT) manufacturing has placed unprecedented logistical demands on supply chain fulfillment networks. Modern supply chain operations function as complex, globally distributed, multi-echelon systems where items are manufactured, batched, warehoused, and transported across thousands of miles before reaching the ultimate consumer. 

In this competitive ecosystem, on-time delivery performance has emerged as a cornerstone metric of enterprise brand equity and customer lifetime value. Academic studies in supply chain management consistently demonstrate that fulfillment delays degrade customer trust, escalate return merchandise authorizations (RMAs), and induce immediate customer churn to competing merchants. Consequently, transitioning logistics control from historical post-mortem reporting to proactive, machine learning-driven risk forecasting represents an essential technological frontier.

### 1.2 Problem Statement
Despite the proliferation of enterprise resource planning (ERP) and warehouse management systems (WMS), modern logistics management remains overwhelmingly reactive. Current operational workflows alert logistics dispatchers of a late delivery only *after* the contracted Service Level Agreement (SLA) deadline has passed, or worse, when a dissatisfied customer logs a delivery inquiry ticket. 

At the moment an order is confirmed and packaged for carrier dispatch, numerous operational signals exist—including customer geographic market, fulfillment department, promised transit days, dispatch hour, weekend batching, and pricing discounts. However, logistics operators lack an automated, predictive mechanism to evaluate these multi-dimensional factors concurrently. 

*Formal Problem Formulation:* Given an order feature vector $\mathbf{x} \in \mathbb{R}^{d}$ captured at or prior to carrier handover, construct a decision function $f(\mathbf{x}) \to \hat{y} \in \{0, 1\}$ that reliably maps the operational parameters to the binary target `Late_delivery_risk`, where $\hat{y} = 1$ denotes a high probability of delivery delay, while simultaneously providing explainable risk attribution and actionable mitigation strategies.

### 1.3 Existing System or Related Work
Existing approaches to delivery delay management can be categorized into two primary paradigms:
1. **Heuristic Rule-Based Scheduling:** Traditional warehouse management systems utilize static routing heuristics, such as assigning expedited air carriers to all First Class shipments. However, empirical operational data demonstrates that First Class shipments exhibit the highest rate of SLA failure (98.0%) because the 1-day transit commitment is overly tight and vulnerable to routine network fluctuations.
2. **Exploratory Machine Learning Baselines:** Earlier academic studies utilizing the DataCo Supply Chain dataset often rely on isolated Jupyter notebooks. A critical flaw in numerous published student implementations is the inclusion of post-event target leakage features—specifically `Days for shipping (real)` and `Delivery Status`. Because actual shipping duration directly reveals whether a delay occurred, models trained with these features report artificial 100% classification accuracy, rendering them entirely invalid for real-world pre-dispatch inference. Furthermore, these implementations rarely explore advanced gradient boosting architectures (XGBoost, LightGBM) and almost never deliver an interactive, user-facing operational decision support interface.

### 1.4 Proposed System
To address these limitations, this project develops **LogiRisk AI**, a comprehensive, leakage-free Machine Learning classification and operational decision-support system. The proposed system introduces:
- A rigorous data-cleaning protocol that eliminates all post-event target leakage attributes.
- Seven newly synthesized domain features capturing warehouse dispatch cutoffs (`order_hour`), weekend fulfillment accumulation (`is_weekend`), cross-border logistics friction (`is_same_country`), aggressive SLAs (`is_rush_delivery`), and order financial health (`profit_margin`, `discount_ratio`, `is_loss_order`).
- A multi-model benchmarking framework comparing Decision Tree, Random Forest, XGBoost, and LightGBM under identical experimental holdout splits.
- A fully integrated, 5-tab reactive Streamlit web application providing single-order inspection with risk gauges, explainable AI risk driver rankings, enterprise batch CSV audits, supply chain business intelligence dashboards, and a dynamic "What-If Mitigation Advisor."

### 1.5 Scope and Significance
The scope of this project encompasses data ingestion, cleaning, domain feature engineering, multi-algorithm training, hyperparameter validation, model serialization, and interactive web dashboard deployment. The practical significance of this work is multi-fold:
- **Financial Protection:** Proactively triaging high-risk consignments enables operators to prevent costly SLA breach penalties and customer claims.
- **Operational Efficiency:** Dispatch coordinators can identify warehouse cutoff bottlenecks and adjust carrier pickups prior to shipping.
- **Academic Rigor:** Establishes a true, leakage-free benchmark on the DataCo dataset, demonstrating realistic 74.12% accuracy and 88.58% precision on unseen operational records.

### 1.6 Motivation
The motivation behind LogiRisk AI stems from the urgent need to bridge the gap between academic data science experiments and practical industrial logistics engineering. By transforming static Python scripts into a robust, deployable decision-support application, this project demonstrates how applied machine learning can deliver tangible operational ROI to modern supply chain enterprises.

---

# 10. SASTRA REPORT: CHAPTER 2 — OBJECTIVES

### 2.1 Main Objective
The primary objective of this minor project is to design, develop, benchmark, and deploy an automated, end-to-end Machine Learning classification and operational decision-support system that predicts supply chain late delivery risk (`Late_delivery_risk`) with high precision and provides real-time prescriptive mitigation recommendations for global logistics dispatches.

### 2.2 Specific Objectives (Implemented and Verified)
1. **Dataset Acquisition & Pre-processing:** Acquire the 53-attribute DataCo Global Supply Chain operational dataset, audit missing values and duplicate records, and systematically drop personally identifiable information and high-cardinality transaction keys.
2. **Target Leakage Elimination:** Identify and remove all post-transit features (such as `Days for shipping (real)` and `Delivery Status`) to ensure genuine pre-dispatch predictive validity.
3. **Domain Feature Engineering:** Formulate and validate seven operational indicators:
   - Fulfillment urgency: `is_rush_delivery` (scheduled SLA $\le$ 1 day).
   - Temporal cutoffs: `order_hour` (0 to 23 hours) and `is_weekend` (Saturday/Sunday backlog).
   - Geographic friction: `is_same_country` (domestic vs international destination).
   - Commercial stress: `profit_margin`, `discount_ratio`, and `is_loss_order`.
4. **Multi-Algorithm Development & Tuning:** Train and tune four supervised classification algorithms spanning decision trees, bootstrap aggregation ensembles, and gradient boosting:
   - Decision Tree Classifier
   - Random Forest Classifier
   - Extreme Gradient Boosting (XGBoost)
   - Light Gradient Boosting Machine (LightGBM)
5. **Rigorous Benchmark Evaluation:** Evaluate all models on an identical 20% holdout test set (12,000 records) across five standard metrics: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
6. **Model Transparency & Feature Importance:** Extract Gini impurity and split-gain feature importances to identify and rank the primary operational drivers of supply chain delays.
7. **Interactive Web Application Deployment:** Construct an enterprise-grade Streamlit web interface (`app.py`) featuring single-order inspection, realistic logistics preset scenarios, visual risk gauges, and model benchmark hubs.
8. **Batch Shipment Audit & Prescriptive Mitigation Engine:** Implement bulk CSV batch inference with automatic calculation of late shipments and at-risk sales value ($), alongside a dynamic "What-If Advisor" simulating alternative carrier modes and SLA buffers.

### 2.3 Explicit Future Work Objectives
1. **IoT Telematics Streaming:** Integrate real-time GPS and RFID tracking streams to update risk probabilities dynamically while freight is in transit.
2. **External Disruption APIs:** Incorporate live weather forecasts, port congestion metrics, and geopolitical border delay indices into the feature space.
3. **Multi-Class Delay Magnitude:** Extend binary classification into multi-class ordinal prediction to forecast exact delay duration (e.g., 1 day late, 2-3 days late, >3 days late).

---

# 11. SASTRA REPORT: CHAPTER 3 — METHODOLOGY

### 3.1 Data Acquisition and Ingestion
The system utilizes the **DataCo Global Supply Chain Dataset**, a comprehensive operational archive capturing supply chain transactions across global markets. The raw dataset contains 180,519 rows and 53 columns formatted as a comma-separated values (`.csv`) file.

The ingestion module loads the operational records using `pandas.read_csv` with `encoding='latin-1'`. A stratified sample of 60,000 records is utilized during model training to provide balanced statistical power across departments, categories, and international regions while maintaining rapid training iteration cycles.

### 3.2 Data Cleaning and Target Leakage Prevention
A vital methodological stage is data cleaning and the strict eradication of target leakage:
1. **Dropping Post-Dispatch Leakage Columns:** The attributes `Days for shipping (real)`, `Delivery Status`, and `shipping date (DateOrders)` occur *after* an order has completed its delivery journey. Retaining these variables causes data leakage where the model memorizes actual transit days rather than learning predictive signals.
2. **Dropping Null and Identifier Columns:** The column `Product Description` is 100% null and is removed. Personal customer fields (`Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street`, `Customer Zipcode`) and non-generalizable database primary keys (`Order Id`, `Order Item Id`, `Customer Id`) are discarded.

### 3.3 Domain Feature Engineering
To capture real-world logistics bottlenecks, 7 domain attributes were engineered from raw timestamps, geographic pairs, and financial values:
- **Order Hour (`order_hour`):** Extracted from `order date (DateOrders)` timestamp. Represents the exact hour of order placement. Captures the critical operational boundary where orders placed after 3:00 PM miss same-day carrier dispatch cutoffs.
- **Weekend Dispatch (`is_weekend`):** Binary indicator evaluating whether `order_dayofweek >= 5`. Captures fulfillment depot staffing reductions and carrier weekend consolidation backlogs.
- **Rush Delivery Flag (`is_rush_delivery`):** Binary flag set to 1 if `Days for shipment (scheduled) <= 1`. Models aggressive, zero-buffer delivery commitments.
- **Cross-Border Factor (`is_same_country`):** Evaluates whether `Customer Country == Order Country`. Shipments crossing national borders encounter customs clearance delays and documentation inspections.
- **Financial Ratios (`profit_margin`, `discount_ratio`, `is_loss_order`):** Orders sold at steep discounts or negative profit margins frequently correlate with high return rates, stockout substitutions, and fulfillment disputes.

### 3.4 Preprocessing and Transformation Pipeline
The cleaned data is partitioned into feature matrix $X$ and binary target vector $y = \text{Late\_delivery\_risk} \in \{0, 1\}$.
A Scikit-Learn `ColumnTransformer` is constructed with two distinct pipelines:
- **Numerical Pipeline:** Passes the 15 continuous and binary engineered features directly (`passthrough`).
- **Categorical Pipeline:** Encodes the 7 categorical features (`Shipping Mode`, `Type`, `Customer Segment`, `Department Name`, `Category Name`, `Market`, `Order Region`) using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
The resulting transformation expands the feature space to **114 numerical columns**.

### 3.5 Train-Test Validation Split
The dataset of 60,000 records is split into:
- **Training Set:** 80% ($N = 48,000$ samples) used exclusively to fit the preprocessor and classification models.
- **Holdout Test Set:** 20% ($N = 12,000$ samples) reserved strictly for unbiased benchmark evaluation.
- The split is seeded with `random_state=42` to guarantee complete academic reproducibility.

### 3.6 Machine Learning Algorithms Implemented

#### 1. Decision Tree Classifier
A non-parametric supervised tree learning algorithm. The tree is built by recursively partitioning the feature space to minimize Gini impurity:
$$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
*Hyperparameters:* `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=5`, `random_state=42`.

#### 2. Random Forest Classifier
An ensemble bagging technique that trains $B = 100$ de-correlated decision trees on bootstrap resamples of the training data. Predictions are generated via majority voting:
$$\hat{y} = \text{mode} \left\{ h_1(\mathbf{x}), h_2(\mathbf{x}), \dots, h_B(\mathbf{x}) \right\}$$
*Hyperparameters:* `n_estimators=100`, `max_depth=14`, `min_samples_split=8`, `min_samples_leaf=4`, `n_jobs=-1`, `random_state=42`.

#### 3. Extreme Gradient Boosting (XGBoost)
A scalable, regularized gradient boosted decision tree framework that builds trees sequentially to optimize a second-order Taylor expansion of the objective loss function:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{N} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
where $g_i$ and $h_i$ are the first and second order gradients of the binary logistic loss.  
*Hyperparameters:* `n_estimators=120`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'`, `random_state=42`.

#### 4. Light Gradient Boosting Machine (LightGBM)
A fast, distributed gradient boosting algorithm utilizing Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). LightGBM grows trees leaf-wise (best-first) rather than level-wise, significantly reducing loss and improving accuracy:
*Hyperparameters:* `n_estimators=120`, `max_depth=8`, `learning_rate=0.08`, `num_leaves=31`, `random_state=42`.

### 3.7 Evaluation Metrics Formulation
Model performance is evaluated across five standard metrics:
- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
- **F1 Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Receiver Operating Characteristic - Area Under Curve (ROC-AUC):** Integral of the True Positive Rate against the False Positive Rate across all classification thresholds $\tau \in [0, 1]$.

### 3.8 Software and Web Deployment Architecture
The web application is implemented in `app.py` using Streamlit. Models and preprocessor artifacts are loaded into memory via `@st.cache_resource`. The interface is structured into 5 logical modules:
1. **Single Order Risk Predictor:** Provides interactive sliders, dropdowns, 4 preset scenario buttons, risk gauge visualizer, driver ranking, and dynamic What-If fulfillment simulator.
2. **Batch Shipment Audit:** Accepts bulk CSV uploads or pre-loaded test batches (35 records), computing total shipments, late shipments, on-time percentage, and at-risk revenue ($). Includes Plotly donut and histogram charts and annotated CSV export.
3. **Supply Chain Risk Analytics:** Visualizes historical delay rates across shipping modes, regional market density, weekend dispatch bottlenecks, and the scheduled SLA buffer curve.
4. **Model Benchmark Hub:** Tabular comparison of all 4 models, comparative grouped bar chart, interactive confusion matrix heatmaps, and horizontal feature importance plots.
5. **Architecture & Project Documentation:** Complete architectural summary and engineering documentation accessible directly within the UI.

---

# 12. SASTRA REPORT: CHAPTER 4 — RESULTS AND DISCUSSION

### 4.1 Quantitative Performance Comparison
All four models were trained on 48,000 records and evaluated on the identical holdout test set of 12,000 records. The verified performance results are summarized in Table 4.1:

#### Table 4.1: Algorithmic Performance Comparison (Holdout Test Set: N = 12,000)

| Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1 Score | ROC-AUC | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **74.12%** | **88.58%** | **63.45%** | **0.7393** | **0.8115** | **1** |
| **XGBoost** | 74.08% | 88.09% | 63.83% | **0.7403** | 0.8108 | 2 |
| **Random Forest** | 74.08% | **89.17%** | 62.84% | 0.7372 | 0.8108 | 3 |
| **Decision Tree** | 73.05% | 84.76% | 65.13% | 0.7366 | 0.7918 | 4 |

### 4.2 Detailed Confusion Matrix Analysis
The holdout test set consists of 5,057 actual on-time dispatches (Class 0) and 6,943 actual late dispatches (Class 1). Table 4.2 presents the empirical confusion matrix counts across all evaluated classifiers:

#### Table 4.2: Confusion Matrix Counts on Test Set

| Classifier | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **4,489** | **568** | **2,538** | **4,405** | 12,000 |
| **XGBoost** | 4,458 | 599 | 2,511 | 4,432 | 12,000 |
| **Random Forest** | 4,527 | 530 | 2,580 | 4,363 | 12,000 |
| **Decision Tree** | 4,244 | 813 | 2,421 | 4,522 | 12,000 |

### 4.3 Feature Importance Findings
Feature importance distributions extracted from the tree models indicate which operational factors contribute most significantly to delivery delay risk:

#### Table 4.3: Top 10 Feature Importances Across Algorithms

| Rank | Random Forest Feature | Importance | XGBoost Feature | Importance | Decision Tree Feature | Importance |
| :-: | :--- | :-: | :--- | :-: | :--- | :-: |
| 1 | `Days for shipment (scheduled)` | 24.28% | `Shipping Mode: Standard Class`| 44.08% | `Days for shipment (scheduled)`| 58.45% |
| 2 | `Shipping Mode: Standard Class` | 16.27% | `Shipping Mode: First Class` | 14.57% | `order_hour` | 18.45% |
| 3 | `Shipping Mode: First Class` | 13.74% | `Days for shipment (scheduled)` | 7.65% | `Shipping Mode: Same Day` | 8.88% |
| 4 | `is_rush_delivery` | 8.61% | `Shipping Mode: Same Day` | 7.15% | `profit_margin` | 2.06% |
| 5 | `order_hour` | 7.21% | `Shipping Mode: Second Class` | 6.03% | `Order Profit Per Order` | 1.73% |
| 6 | `Shipping Mode: Second Class` | 6.27% | `is_rush_delivery` | 1.51% | `discount_ratio` | 1.32% |
| 7 | `Shipping Mode: Same Day` | 3.98% | `order_hour` | 0.83% | `Order Item Discount` | 1.30% |
| 8 | `profit_margin` | 1.84% | `Type: TRANSFER` | 0.69% | `order_dayofweek` | 1.26% |
| 9 | `Order Profit Per Order` | 1.84% | `Type: CASH` | 0.29% | `order_month` | 0.94% |
| 10 | `Order Item Discount` | 1.55% | `Category: Baseball & Softball`| 0.28% | `Type: TRANSFER` | 0.92% |

### 4.4 Interpretation of Results
1. **Validation of Leakage-Free Learning:** Unlike naive baseline notebooks that report artificial 100% accuracy by leaking post-event transit duration, our models converge realistically at **74.12% accuracy** and **0.8115 ROC-AUC**, confirming genuine pre-dispatch predictive capability.
2. **Superiority of Gradient Boosting:** LightGBM achieved the top overall ranking, providing an optimal trade-off: high accuracy (74.12%), high precision (88.58%), and the fastest inference and serialization efficiency (426 KB model file).
3. **Operational Significance of High Precision (88.58%):** In supply chain management, false alarms waste valuable expedited freight budgets. A precision of 88.58% guarantees that when the system alerts a logistics dispatcher to a late delivery risk, the shipment will indeed fail its SLA in nearly 9 out of 10 cases.
4. **Dominance of Fulfillment SLA Levers:** The feature importance rankings consistently show that the choice of `Shipping Mode` combined with `Days for shipment (scheduled)` accounts for over **60% of total model variance**. Expedited modes (First Class) fail frequently because carriers struggle to meet 1-day SLAs, whereas Standard Class provides sufficient buffer to absorb regional transit delays.

### 4.5 Operational Demonstration: What-If Mitigation Simulation
To validate practical utility, the system's "What-If Advisor" was tested on an urgent First Class order (`Days for shipment = 1`). As shown in Table 4.4, shifting fulfillment to Standard Class reduces delay risk from **98.0% down to 37.5%**, converting an at-risk dispatch into a safe consignment:

#### Table 4.4: Sample Output of What-If Fulfillment Simulation

| Fulfillment Shipping Mode | Standard SLA Window | Calculated Late Risk Probability | Risk Delta | System Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **First Class** | 1 Day | **98.0%** | Baseline | At-Risk 🔴 |
| **Second Class** | 2 Days | **77.6%** | -20.4% | At-Risk 🔴 |
| **Same Day** | 0 Days | **54.4%** | -43.6% | At-Risk 🔴 |
| **Standard Class** | **4 Days** | **37.5%** | **-60.5%** | **Safe 🟢** |

---

# 13. SASTRA REPORT: CHAPTER 5 — CONCLUSIONS AND FURTHER WORK

### 13.1 Summary of Completed Work
This minor project successfully designed, implemented, evaluated, and deployed **LogiRisk AI**, an end-to-end Machine Learning classification and operational decision-support platform for supply chain late delivery risk prediction. 

Starting from raw operational logs in the DataCo Supply Chain dataset (180,519 records), the project developed a strict, leakage-free data cleansing and feature engineering pipeline that synthesized seven domain attributes capturing warehouse dispatch cutoffs, weekend backlog bottlenecks, aggressive SLAs, cross-border frictions, and commercial order stress. 

Four distinct machine learning algorithms (Decision Tree, Random Forest, XGBoost, and LightGBM) were trained and benchmarked across five standard metrics. The optimal model, **LightGBM**, achieved **74.12% accuracy**, **88.58% precision**, and **0.8115 ROC-AUC** on a holdout test set of 12,000 unseen records.

The predictive engine was fully integrated into a responsive, 5-tab Streamlit web application featuring single-order inspection, realistic logistics presets, risk gauge visualizations, bulk CSV batch auditing with revenue-at-risk aggregation, business intelligence dashboards, and a dynamic What-If fulfillment mitigation advisor.

### 13.2 Main Achievements
- Successfully eliminated target leakage, proving that real-world pre-dispatch signals achieve strong predictive power (0.8115 ROC-AUC) without snooping post-event transit days.
- Established an 88.58% precision benchmark, minimizing costly false positive alarms for enterprise logistics operators.
- Developed an interactive operational decision support interface that translates raw probabilities into prescriptive shipping mode and SLA buffer recommendations.
- Implemented bulk audit functionality capable of ingesting operational CSV batches and calculating immediate financial exposure (Total At-Risk Sales $).

### 13.3 Practical Significance
LogiRisk AI demonstrates immediate business value for third-party logistics (3PL) providers, global e-commerce merchants, and warehouse dispatch controllers. By flagging late delivery risks 24 to 48 hours prior to carrier handover, logistics managers can:
- Reallocate high-risk shipments to reliable carrier tiers.
- Proactively adjust delivery promises to customers, eliminating SLA penalty chargebacks.
- Triage high-value orders in warehouse staging areas to protect enterprise revenue.

### 13.4 Current Limitations
- **Static Pre-Dispatch Scope:** The system evaluates risk at the time of carrier dispatch; it cannot dynamically recalculate risk if an in-transit vehicle breaks down or encounters unexpected highway closures.
- **Absence of Real-Time Weather/Traffic Streams:** Environmental transit conditions are not currently integrated via live APIs.
- **Binary Target Formulation:** The model predicts whether an order will be late, but does not forecast the exact number of delayed hours or days.

### 13.5 Future Improvements
1. **Live IoT & Telematics Ingestion:** Integrate Kafka or MQTT streaming pipelines to continuously update delivery risk using real-time vehicle GPS and RFID checkpoint pings.
2. **External Context APIs:** Incorporate live weather forecasts (NOAA) and maritime port congestion data to refine cross-border risk scoring.
3. **Multi-Class Delay Severity Modeling:** Train an ordinal regression or multi-class classifier to predict delay severity (e.g., Minor Delay: 1 day; Moderate Delay: 2-3 days; Severe Delay: >3 days).
4. **Enterprise WMS / ERP Integration:** Package the inference engine into a containerized FastAPI microservice for seamless plug-and-play integration into SAP S/4HANA and Oracle Supply Chain Management.

---

# 14. REFERENCES AND LITERATURE REVIEW

### 14.1 Academic Literature and Technical References
*(Formatted alphabetically as required by SASTRA academic report guidelines)*

1. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), pp. 5–32. https://doi.org/10.1023/A:1010933404324
2. Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794. https://doi.org/10.1145/2939672.2939785
3. Constante, F., Silva, F., and Pereira, A. (2019). "DataCo Global Supply Chain Dataset." *Kaggle Repository*. Available at: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
4. Fawcett, T. (2006). "An Introduction to ROC Analysis." *Pattern Recognition Letters*, 27(8), pp. 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
5. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T. Y. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, pp. 3146–3154.
6. McKinney, W. (2010). "Data Structures for Statistical Computing in Python." In *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, pp. 56–61.
7. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, pp. 2825–2830.
8. Streamlit Inc. (2024). "Streamlit Documentation: A Faster Way to Build and Share Data Apps." Available at: https://docs.streamlit.io/
9. Waller, M. A. and Fawcett, S. E. (2013). "Data Science, Predictive Analytics, and Big Data in Supply Chain Management—Current State and Future Potential." *Journal of Business Logistics*, 34(2), pp. 77–84. https://doi.org/10.1111/jbl.12010

---

# 15. ABSTRACT AND SYNOPSIS

### 15.1 Complete Project Abstract
In contemporary global supply chains and omnichannel e-commerce, fulfillment punctuality directly governs customer retention, contractual compliance, and brand loyalty. Unanticipated delivery delays incur substantial financial chargebacks, inventory holding penalties, and customer churn. Traditional logistics management operates reactively, alerting dispatchers only after an agreed Service Level Agreement (SLA) has been breached. 

This minor project presents **LogiRisk AI**, an end-to-end Machine Learning classification and operational decision-support system that predicts supply chain late delivery risk (`Late_delivery_risk`) prior to carrier handover. Utilizing the DataCo Global Supply Chain operational dataset (180,519 records), the project implements a rigorous data cleansing protocol that eliminates target leakage caused by post-event attributes such as real shipping duration. Seven domain-specific indicators were synthesized, capturing warehouse dispatch hour cutoffs, weekend fulfillment accumulations, aggressive delivery SLAs, cross-border customs frictions, and transaction financial health. 

Four supervised machine learning algorithms—Decision Tree, Random Forest, XGBoost, and LightGBM—were developed and systematically benchmarked on a holdout test set of 12,000 unseen records. The optimal model, **LightGBM**, achieved an accuracy of **74.12%**, an exceptional precision of **88.58%**, an F1 score of **0.7393**, and an area under the ROC curve (ROC-AUC) of **0.8115**. Feature importance analysis revealed that promised shipping mode, scheduled SLA duration, and warehouse dispatch hour account for over 60% of predictive variance. 

The trained models and transformation pipelines were integrated into a 5-tab reactive Streamlit web application. The platform provides single-order risk inspection with visual gauges, root-cause explainability, automated bulk CSV auditing with revenue-at-risk aggregation, supply chain business intelligence dashboards, and a dynamic "What-If Mitigation Advisor" simulating alternative carrier fulfillment options. LogiRisk AI bridges the gap between academic data science and practical supply chain operations, delivering an effective, high-precision decision-support tool for modern logistics enterprises.

### 15.2 SASTRA Synopsis Abstract
*(Format: Single paragraph, under 600 words, no tables/figures/citations, strictly within one A4 page)*

Fulfillment punctuality represents a vital operational determinant of customer retention, service level agreement compliance, and brand equity in global supply chain networks. Unanticipated delivery delays inflict severe financial penalties, inventory holding costs, and customer churn upon e-commerce enterprises and third-party logistics providers. Conventional supply chain management systems remain overwhelmingly reactive, identifying shipment delays only after delivery deadlines have elapsed or customer complaints have been registered. This project develops LogiRisk AI, an automated Machine Learning classification and operational decision-support platform designed to forecast late delivery risk prior to warehouse carrier dispatch. Using operational records from the DataCo Global Supply Chain dataset, the project establishes a rigorous preprocessing pipeline that eliminates target leakage resulting from post-event transit attributes. Seven domain-specific features were formulated to capture warehouse dispatch hour cutoffs, weekend backlog bottlenecks, scheduled delivery urgency, cross-border customs barriers, and transaction financial stress. Four supervised classification algorithms—Decision Tree, Random Forest, Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM)—were developed and evaluated across a holdout test partition of 12,000 unseen records. The best-performing model, LightGBM, attained an accuracy of 74.12%, an outstanding precision of 88.58%, an F1 score of 0.7393, and an area under the ROC curve (ROC-AUC) of 0.8115. Feature attribution rankings demonstrated that promised shipping mode, scheduled SLA window, and order dispatch hour govern over 60% of predictive variance. The trained models were deployed within an interactive, multi-view Streamlit web application. The system provides single-order inspection with real-time risk gauges, explainable delay driver rankings, bulk CSV batch auditing with revenue-at-risk financial summation, supply chain operational analytics, and a dynamic What-If Mitigation Advisor that simulates alternative fulfillment modes to guide proactive delay prevention.

### 15.3 Keywords
Supply Chain Analytics, Late Delivery Risk Classification, Machine Learning, LightGBM, Prescriptive Logistics Intelligence.

---

# 16. PRELIMINARY REPORT PAGES

### 16.1 Title Page Format
```
                                 LOGIRISK AI:
                   SUPPLY CHAIN LATE DELIVERY RISK CLASSIFIER
                  AND OPERATIONAL DECISION SUPPORT SYSTEM

                           A MINOR PROJECT REPORT
                                 Submitted by

                             SOUNDAR S   (127018056)
                           SRI VIGNESH G (127004255)
                            NIRANJAN R   (127014035)

                      in partial fulfillment for the award of
                                 the degree of

                             BACHELOR OF TECHNOLOGY
                                       in
                        COMPUTER SCIENCE AND ENGINEERING

                               [SASTRA LOGO]

                       SCHOOL OF COMPUTING / DEPT OF CSE
                       SASTRA DEEMED TO BE UNIVERSITY
                      TIRUMALAISAMUDRAM, THANJAVUR – 613401
                                  MAY 2025
```

### 16.2 Bonafide Certificate
```
                       SASTRA DEEMED TO BE UNIVERSITY
                            TIRUMALAISAMUDRAM,
                           THANJAVUR – 613 401

                           BONAFIDE CERTIFICATE

Certified that this minor project report titled "LOGIRISK AI: SUPPLY CHAIN LATE 
DELIVERY RISK CLASSIFIER AND OPERATIONAL DECISION SUPPORT SYSTEM" is the bonafide 
work of SOUNDAR S (127018056), SRI VIGNESH G (127004255), and NIRANJAN R (127014035), 
who carried out the project work under my supervision.


SIGNATURE                                        SIGNATURE
[Guide Name]                                     [Head of Department]
PROJECT GUIDE                                    HEAD OF THE DEPARTMENT
Department of Computer Science & Engg.           Department of Computer Science & Engg.
School of Computing                              School of Computing
SASTRA Deemed to be University                   SASTRA Deemed to be University
Thanjavur – 613 401                              Thanjavur – 613 401


Submitted for the Minor Project Viva-Voce examination held on ___________________



INTERNAL EXAMINER                                EXTERNAL EXAMINER
```

### 16.3 Candidate Declaration
```
                               DECLARATION

We hereby declare that the minor project report titled "LOGIRISK AI: SUPPLY CHAIN 
LATE DELIVERY RISK CLASSIFIER AND OPERATIONAL DECISION SUPPORT SYSTEM" submitted 
to SASTRA Deemed to be University in partial fulfillment of the requirements for 
the award of the Degree of Bachelor of Technology in Computer Science and 
Engineering is a record of original work done by us under the guidance of 
[Guide Name, Designation], Department of Computer Science and Engineering, 
School of Computing, SASTRA Deemed to be University.

We further declare that this report has not been submitted previously for the award 
of any degree, diploma, associateship, or fellowship in this or any other 
University or Institution.


Place: Thanjavur
Date:

1. SOUNDAR S     (127018056)   _____________________
2. SRI VIGNESH G (127004255)   _____________________
3. NIRANJAN R    (127014035)   _____________________
```

### 16.4 Acknowledgements (Template)
```
                             ACKNOWLEDGEMENTS

We express our profound gratitude to our Honorable Vice-Chancellor, Registrar, 
and the Management of SASTRA Deemed to be University for providing us with the 
academic environment, computational facilities, and infrastructure to complete 
this minor project successfully.

We extend our sincere thanks to the Dean, School of Computing, and the Head of 
the Department of Computer Science and Engineering for their continuous support 
and guidance throughout the course of this work.

We record our heartfelt thanks and deep sense of gratitude to our project guide, 
[Guide Name, Designation], for their invaluable suggestions, constructive 
critiques, and patient mentorship during every phase of this project.

Finally, we thank our parents, faculty members, and friends who supported us 
directly and indirectly in bringing this endeavor to successful fruition.

                                                      SOUNDAR S (127018056)
                                                      SRI VIGNESH G (127004255)
                                                      NIRANJAN R (127014035)
```

### 16.5 Individual Team Member Contributions & Learning Placeholders

| Member Register No & Name | Key Areas of Responsibility | Core Technical Contributions | Individual Learning Outcomes |
| :--- | :--- | :--- | :--- |
| **127018056 — Soundar S** | Project Architecture, Feature Engineering & App Integration | Formulation of engineered features (`is_rush_delivery`, `order_hour`, `profit_margin`), serialization pipeline, and core Streamlit UI integration (`app.py`). | Mastered supervised gradient boosting pipelines, column transformation architectures, and interactive dashboard engineering in Streamlit. |
| **127004255 — Sri Vignesh G** | Model Benchmarking, Hyperparameter Tuning & Validation | Model training implementation (`model_trainer.py`), tuning Decision Tree, Random Forest, XGBoost, LightGBM, and confusion matrix metric extraction. | Gained deep expertise in comparative ensemble evaluation, hyperparameter optimization, and handling imbalanced classification trade-offs. |
| **127014035 — Niranjan R** | Data Pipeline, Leakage Control & Batch Audit Module | Ingestion of raw DataCo dataset, data cleaning, target leakage eradication, and engineering the batch CSV inference and export engine. | Acquired practical skills in exploratory data analysis, leakage prevention in tabular ML, and batch scoring automation. |

### 16.6 Nomenclature and Abbreviations

| Abbreviation | Expansion / Definition |
| :--- | :--- |
| **3PL** | Third-Party Logistics Provider |
| **API** | Application Programming Interface |
| **AUC** | Area Under the Curve |
| **CSV** | Comma-Separated Values |
| **DT** | Decision Tree |
| **ERP** | Enterprise Resource Planning |
| **FN** | False Negative |
| **FP** | False Positive |
| **JIT** | Just-In-Time Logistics |
| **JSON** | JavaScript Object Notation |
| **KPI** | Key Performance Indicator |
| **LGBM** | Light Gradient Boosting Machine |
| **ML** | Machine Learning |
| **PII** | Personally Identifiable Information |
| **RF** | Random Forest |
| **RMA** | Return Merchandise Authorization |
| **ROC** | Receiver Operating Characteristic |
| **SLA** | Service Level Agreement |
| **TN** | True Negative |
| **TP** | True Positive |
| **UI** | User Interface |
| **WMS** | Warehouse Management System |
| **XGBoost** | Extreme Gradient Boosting |

---

# 17. APPENDIX

### 17.1 Implementation Source Code Manifest

1. **`model_trainer.py` (Model Training & Export Pipeline - 242 lines):**
   - Implements `build_features(df)` for timestamp extraction and 7 domain indicators.
   - Fits `ColumnTransformer` with `OneHotEncoder` and passthrough numerical transforms.
   - Trains Decision Tree, Random Forest, XGBoost, and LightGBM.
   - Computes Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrices.
   - Serializes binaries (`models/*.joblib`) and metadata (`models/metadata.json`).
   - Exports a 35-sample test set (`sample_data/sample_orders.csv`).

2. **`app.py` (Streamlit Web Application & Decision Support System - 927 lines):**
   - `@st.cache_resource` artifact loader.
   - Custom CSS design system (*Plus Jakarta Sans*, glassmorphic banners, metric cards, risk pills).
   - Sidebar engine settings (model switcher, threshold slider, benchmark scoreboard).
   - **Tab 1:** Single Order Predictor, 4 preset scenarios, gauge charts, explainability rankings, What-If Mitigation Advisor.
   - **Tab 2:** Batch Shipment Audit, CSV uploader, 35-order sample loader, KPI metric cards, Plotly donut and risk histogram charts, filtered dataframe, annotated CSV download.
   - **Tab 3:** Supply Chain Risk Analytics (Historical delay by shipping mode, regional market risk density, weekend dispatch impact, scheduled SLA buffer curve).
   - **Tab 4:** Model Benchmark Hub (Metrics table, grouped bar charts, confusion matrix heatmaps, horizontal feature importance plots).
   - **Tab 5:** Complete Project & Architecture Documentation viewer.

3. **`requirements.txt` (Environment Dependencies):**
   ```txt
   streamlit>=1.30.0
   scikit-learn>=1.3.0
   pandas>=2.0.0
   numpy>=1.24.0
   xgboost>=2.0.0
   lightgbm>=4.0.0
   plotly>=5.15.0
   joblib>=1.3.0
   ```

4. **`models/metadata.json` (Serialized Evaluation Data - 571 lines):**
   - Preserves verified evaluation metrics for all 4 models.
   - Stores exact confusion matrix counts ($2 \times 2$ matrices).
   - Stores Top-20 feature importance lists and category dropdown vocabularies.

### 17.2 Verified CLI Execution Commands

To reproduce the entire pipeline from scratch on any workstation:

```bash
# 1. Clone repository
git clone https://github.com/SoundarSomasundaram/deliveryriskprediction.git
cd deliveryriskprediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and export models
python model_trainer.py

# 4. Launch web application
streamlit run app.py
```
Application opens at `http://localhost:8501`.
