# 🚚 LogiRisk AI — Supply Chain Late Delivery Risk Classifier

LogiRisk AI is an end-to-end Machine Learning and decision-support web application that predicts **Late Delivery Risk** (`Late_delivery_risk`) in global supply chain logistics.

Built upon the **DataCo Global Supply Chain Dataset**, this solution enhances the initial notebook baseline by introducing advanced domain feature engineering, multi-model benchmarking (Decision Tree, Random Forest, XGBoost, LightGBM), and an interactive **Streamlit UI**.

---

## 🌟 Key Features

1. **🔮 Single Order Risk Prediction & Simulator**:
   - Interactive input form for logistics, product details, geographic market, order timing, and financial parameters.
   - Quick-load preset scenarios (*Urgent First Class Rush*, *Safe Standard Ground*, *Cross-Border Priority*, *Weekend Heavy Order*).
   - Real-time gauge and probability meter with risk categorization (Low 🟢, Moderate 🟡, High 🔴).
   - **Explainable AI (XAI)**: Identifies primary risk drivers (e.g. tight SLA, weekend dispatch, cross-border shipping).
   - **What-If Mitigation Advisor**: Automatically simulates how altering shipping mode or scheduled SLA days lowers risk.

2. **📂 Batch Shipment Audit & CSV Inference**:
   - Upload CSV batches for bulk inference or load pre-packaged sample test orders with one click.
   - Real-time KPI summaries: Total Orders, Flagged Late Shipments, % On-Time, and At-Risk Sales Value ($).
   - Interactive table with risk filtering and one-click annotated CSV export.

3. **📊 Supply Chain Risk Analytics**:
   - Delay rate breakdown by Shipping Mode vs. Scheduled SLA days.
   - Regional delay distribution across 5 global markets (Europe, USCA, Pacific Asia, LATAM, Africa).
   - Weekend fulfillment bottleneck analysis and SLA buffer curves.

4. **🏆 Model Benchmark Hub**:
   - Comparative evaluation of Decision Tree, Random Forest, XGBoost, and LightGBM across Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
   - Interactive Confusion Matrix viewer and Top 20 Feature Importance leaderboard.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Re-train & Export Models
The models are already trained and exported into the `models/` folder. To re-train or update features:
```bash
python model_trainer.py
```

### 3. Launch the Streamlit Web UI
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the application!

---

## 🔬 Engineered Features

- **`is_rush_delivery`**: Delivery scheduled within 1 day or less.
- **`is_weekend`**: Order placed on Saturday or Sunday.
- **`order_hour`**: Hour of the day the order was placed (captures daily dispatch cutoffs).
- **`is_same_country`**: Flags domestic vs. cross-border international shipments.
- **`profit_margin`**: Ratio of net profit to total sales value.
- **`discount_ratio`**: Ratio of discount applied to total order value.
- **`is_loss_order`**: Binary indicator for loss-making order lines.

---

## 🤖 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **74.12%** | **88.58%** | **63.45%** | **0.7393** | **0.8115** |
| **XGBoost** | **74.08%** | **88.09%** | **63.83%** | **0.7403** | **0.8108** |
| **Random Forest** | 74.08% | 89.17% | 62.84% | 0.7372 | 0.8108 |
| **Decision Tree** | 73.05% | 84.76% | 65.13% | 0.7366 | 0.7918 |
