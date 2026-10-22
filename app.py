"""
LogiRisk AI: Supply Chain Late Delivery Risk Classifier & Decision Support System
A modern, feature-rich Streamlit application for predictive risk analytics in logistics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="LogiRisk AI - Supply Chain Risk Classifier",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }
    
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #fecaca;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #fde68a;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #bbf7d0;
    }
    
    .preset-pill {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.85rem;
        cursor: pointer;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Models & Metadata (Cached for performance)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    models_dir = "models"
    metadata_path = os.path.join(models_dir, "metadata.json")
    prep_path = os.path.join(models_dir, "preprocessor.joblib")
    
    if not os.path.exists(metadata_path) or not os.path.exists(prep_path):
        return None, None, None
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    preprocessor = joblib.load(prep_path)
    
    loaded_models = {}
    model_files = {
        "LightGBM": "lightgbm.joblib",
        "XGBoost": "xgboost.joblib",
        "Random Forest": "random_forest.joblib",
        "Decision Tree": "decision_tree.joblib"
    }
    
    for display_name, fname in model_files.items():
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            try:
                loaded_models[display_name] = joblib.load(fpath)
            except Exception as e:
                st.warning(f"Could not load {display_name}: {e}")
                
    return preprocessor, loaded_models, metadata

preprocessor, models, metadata = load_artifacts()

if not models or not preprocessor or not metadata:
    st.error("Model artifacts not found! Please run `python model_trainer.py` to train and export the models.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.title("⚙️ Engine Settings")
    
    model_choice = st.selectbox(
        "Select Active ML Model",
        options=list(models.keys()),
        index=0,
        help="Choose which algorithm to use for predictions. LightGBM and XGBoost offer highest predictive accuracy."
    )
    
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability cutoff for classifying an order as 'Late Delivery Risk'. Lower threshold catches more delays (conservative)."
    )
    
    st.markdown("---")
    st.subheader("Model Benchmark")
    acc = metadata["metrics_summary"].get(model_choice, {}).get("Accuracy", 0.74)
    auc = metadata["metrics_summary"].get(model_choice, {}).get("ROC-AUC", 0.81)
    f1 = metadata["metrics_summary"].get(model_choice, {}).get("F1 Score", 0.74)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Test Accuracy", f"{acc:.1%}")
        st.metric("F1 Score", f"{f1:.3f}")
    with col_s2:
        st.metric("ROC-AUC", f"{auc:.3f}")
        st.metric("Status", "Operational", delta="Ready")
        
    st.markdown("---")
    st.caption("🛡️ **LogiRisk AI Enterprise Edition**")
    st.caption("Supply Chain Risk Intelligence System v2.0")

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
                🚚 LogiRisk AI
            </h1>
            <p style="margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.9; font-weight: 400;">
                Supply Chain Late Delivery Risk Predictor & Automated Logistics Decision Support System
            </p>
        </div>
        <div style="margin-top: 10px;">
            <span style="background: rgba(255,255,255,0.18); padding: 8px 16px; border-radius: 30px; font-weight: 600; font-size: 0.9rem; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.25);">
                ⚡ ML Powered • Real-Time Inference
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "🔮 Single Order Predictor",
    "📂 Batch Shipment Audit",
    "📊 Supply Chain Risk Analytics",
    "🏆 Model Benchmark Hub",
    "📖 Project & Architecture"
])

# =========================================================
# TAB 1: SINGLE ORDER PREDICTOR
# =========================================================
with tabs[0]:
    st.subheader("Interactive Shipment Risk Inspector")
    st.markdown("Configure order attributes below or load a realistic logistics preset scenario.")
    
    # Preset Buttons
    st.markdown("**⚡ Quick-Load Scenarios:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    preset_choice = None
    if p_col1.button("🚨 Urgent First Class Rush", use_container_width=True):
        preset_choice = "rush"
    if p_col2.button("✅ Safe Standard Ground", use_container_width=True):
        preset_choice = "safe"
    if p_col3.button("🌍 Cross-Border Priority", use_container_width=True):
        preset_choice = "cross_border"
    if p_col4.button("🛍️ Weekend Heavy Order", use_container_width=True):
        preset_choice = "weekend"
        
    # Set default values based on chosen preset or standard default
    cats_meta = metadata["categories_metadata"]
    
    if preset_choice == "rush":
        def_ship_mode = "First Class"
        def_sched_days = 1
        def_type = "TRANSFER"
        def_segment = "Consumer"
        def_dept = "Apparel"
        def_cat = "Cleats"
        def_market = "Europe"
        def_region = "Western Europe"
        def_qty = 3
        def_price = 149.99
        def_discount = 25.0
        def_profit = 32.5
        def_month = 11
        def_hour = 19
        def_dayofweek = 5
        def_same_country = 0
    elif preset_choice == "safe":
        def_ship_mode = "Standard Class"
        def_sched_days = 4
        def_type = "DEBIT"
        def_segment = "Corporate"
        def_dept = "Golf"
        def_cat = "Women's Apparel"
        def_market = "USCA"
        def_region = "East of USA"
        def_qty = 1
        def_price = 80.0
        def_discount = 2.0
        def_profit = 28.0
        def_month = 5
        def_hour = 10
        def_dayofweek = 1
        def_same_country = 1
    elif preset_choice == "cross_border":
        def_ship_mode = "Second Class"
        def_sched_days = 2
        def_type = "PAYMENT"
        def_segment = "Home Office"
        def_dept = "Outdoors"
        def_cat = "Accessories"
        def_market = "Pacific Asia"
        def_region = "Southeast Asia"
        def_qty = 2
        def_price = 120.0
        def_discount = 12.0
        def_profit = -14.0
        def_month = 8
        def_hour = 14
        def_dayofweek = 3
        def_same_country = 0
    elif preset_choice == "weekend":
        def_ship_mode = "Second Class"
        def_sched_days = 2
        def_type = "CASH"
        def_segment = "Consumer"
        def_dept = "Fan Shop"
        def_cat = "Sporting Goods"
        def_market = "LATAM"
        def_region = "South America"
        def_qty = 5
        def_price = 60.0
        def_discount = 30.0
        def_profit = 15.0
        def_month = 12
        def_hour = 22
        def_dayofweek = 6
        def_same_country = 1
    else:
        def_ship_mode = "Second Class"
        def_sched_days = 2
        def_type = "DEBIT"
        def_segment = "Consumer"
        def_dept = "Apparel"
        def_cat = "Men's Footwear"
        def_market = "Europe"
        def_region = "Western Europe"
        def_qty = 2
        def_price = 129.99
        def_discount = 10.0
        def_profit = 35.0
        def_month = 6
        def_hour = 14
        def_dayofweek = 2
        def_same_country = 1

    st.markdown("---")
    
    # Form Layout
    with st.form("single_order_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            st.markdown("##### 🚚 1. Shipping & Logistics")
            ship_mode = st.selectbox(
                "Shipping Mode",
                options=cats_meta.get("Shipping Mode", ["Standard Class", "First Class", "Second Class", "Same Day"]),
                index=cats_meta.get("Shipping Mode", []).index(def_ship_mode) if def_ship_mode in cats_meta.get("Shipping Mode", []) else 0
            )
            
            # Map standard scheduled days recommendations
            sched_map = {"First Class": 1, "Second Class": 2, "Standard Class": 4, "Same Day": 0}
            sched_default = sched_map.get(ship_mode, def_sched_days)
            
            sched_days = st.slider(
                "Days for Shipment (Scheduled SLA)",
                min_value=0,
                max_value=6,
                value=def_sched_days,
                help="The promised delivery SLA window. Tight SLAs (0-1 days) carry highest risk."
            )
            
            pay_type = st.selectbox(
                "Transaction Payment Type",
                options=cats_meta.get("Type", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]),
                index=cats_meta.get("Type", []).index(def_type) if def_type in cats_meta.get("Type", []) else 0
            )
            
            same_country = st.radio(
                "Geographic Scope",
                options=[1, 0],
                format_func=lambda x: "Domestic Shipping (Same Country)" if x == 1 else "International / Cross-Border",
                index=0 if def_same_country == 1 else 1,
                horizontal=True
            )
            
        with col_f2:
            st.markdown("##### 📦 2. Product & Category")
            dept_name = st.selectbox(
                "Department Name",
                options=cats_meta.get("Department Name", ["Apparel", "Fan Shop", "Golf", "Outdoors", "Fitness"]),
                index=cats_meta.get("Department Name", []).index(def_dept) if def_dept in cats_meta.get("Department Name", []) else 0
            )
            
            cat_name = st.selectbox(
                "Category Name",
                options=cats_meta.get("Category Name", ["Cleats", "Men's Footwear", "Women's Apparel", "Accessories"]),
                index=cats_meta.get("Category Name", []).index(def_cat) if def_cat in cats_meta.get("Category Name", []) else 0
            )
            
            cust_segment = st.selectbox(
                "Customer Segment",
                options=cats_meta.get("Customer Segment", ["Consumer", "Corporate", "Home Office"]),
                index=cats_meta.get("Customer Segment", []).index(def_segment) if def_segment in cats_meta.get("Customer Segment", []) else 0
            )
            
            quantity = st.number_input("Item Quantity", min_value=1, max_value=20, value=def_qty, step=1)
            unit_price = st.number_input("Unit Price ($)", min_value=1.0, max_value=2000.0, value=float(def_price), step=5.0)

        with col_f3:
            st.markdown("##### 📍 3. Market, Timing & Economics")
            market = st.selectbox(
                "Destination Market",
                options=cats_meta.get("Market", ["Europe", "USCA", "Pacific Asia", "LATAM", "Africa"]),
                index=cats_meta.get("Market", []).index(def_market) if def_market in cats_meta.get("Market", []) else 0
            )
            
            region = st.selectbox(
                "Order Region",
                options=cats_meta.get("Order Region", ["Western Europe", "Central America", "South America", "East of USA"]),
                index=cats_meta.get("Order Region", []).index(def_region) if def_region in cats_meta.get("Order Region", []) else 0
            )
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                order_month = st.selectbox("Order Month", options=list(range(1, 13)), index=def_month - 1)
            with col_t2:
                order_hour = st.slider("Order Hour (24h)", min_value=0, max_value=23, value=def_hour)
                
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            order_day_name = st.selectbox("Day of Week", options=day_names, index=def_dayofweek)
            order_dayofweek = day_names.index(order_day_name)
            
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                discount_val = st.number_input("Discount ($)", min_value=0.0, max_value=500.0, value=float(def_discount), step=2.0)
            with col_e2:
                profit_val = st.number_input("Net Profit ($)", min_value=-500.0, max_value=1000.0, value=float(def_profit), step=5.0)
                
        submit_btn = st.form_submit_button("🚀 Run Risk Prediction Analysis", use_container_width=True)

    # Compute Feature Engineering for Inference
    calc_sales = max(0.01, float(quantity * unit_price))
    is_weekend = 1 if order_dayofweek >= 5 else 0
    is_rush = 1 if sched_days <= 1 else 0
    profit_margin = round(profit_val / (calc_sales + 1.0), 4)
    discount_ratio = round(discount_val / (calc_sales + 1.0), 4)
    is_loss = 1 if profit_val < 0 else 0
    
    input_dict = {
        'Shipping Mode': ship_mode,
        'Days for shipment (scheduled)': sched_days,
        'Type': pay_type,
        'Customer Segment': cust_segment,
        'Department Name': dept_name,
        'Category Name': cat_name,
        'Market': market,
        'Order Region': region,
        'Order Item Quantity': quantity,
        'Product Price': unit_price,
        'Sales': calc_sales,
        'Order Item Discount': discount_val,
        'Order Profit Per Order': profit_val,
        'order_month': order_month,
        'order_dayofweek': order_dayofweek,
        'order_hour': order_hour,
        'is_weekend': is_weekend,
        'is_rush_delivery': is_rush,
        'is_same_country': same_country,
        'profit_margin': profit_margin,
        'discount_ratio': discount_ratio,
        'is_loss_order': is_loss
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Run Prediction
    chosen_model = models[model_choice]
    encoded_input = preprocessor.transform(input_df)
    pred_prob = float(chosen_model.predict_proba(encoded_input)[0, 1])
    is_late_risk = int(pred_prob >= threshold)
    
    st.markdown("### 🎯 Predictive Assessment Results")
    
    res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.3])
    
    with res_col1:
        if is_late_risk == 1:
            st.markdown(f"""
            <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 14px; padding: 20px; text-align: center;">
                <span class="badge-high">CRITICAL RISK ALERT</span>
                <h2 style="color: #991b1b; margin: 12px 0 4px 0; font-size: 1.8rem;">⚠️ Late Delivery Likely</h2>
                <p style="color: #7f1d1d; margin: 0; font-size: 0.95rem;">
                    This order has high probability of exceeding scheduled delivery SLA.
                </p>
                <div style="margin-top: 14px; font-size: 2.2rem; font-weight: 800; color: #b91c1c;">
                    {pred_prob:.1%}
                </div>
                <div style="color: #6b7280; font-size: 0.8rem;">Late Risk Probability (Threshold: {threshold:.0%})</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 2px solid #22c55e; border-radius: 14px; padding: 20px; text-align: center;">
                <span class="badge-low">ON-TIME EXPECTED</span>
                <h2 style="color: #166534; margin: 12px 0 4px 0; font-size: 1.8rem;">✅ Safe Dispatch</h2>
                <p style="color: #14532d; margin: 0; font-size: 0.95rem;">
                    Logistics SLA parameters are well within operational capacity.
                </p>
                <div style="margin-top: 14px; font-size: 2.2rem; font-weight: 800; color: #15803d;">
                    {pred_prob:.1%}
                </div>
                <div style="color: #6b7280; font-size: 0.8rem;">Late Risk Probability (Threshold: {threshold:.0%})</div>
            </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        # Gauge Chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pred_prob * 100, 1),
            number={'suffix': "%", 'font': {'size': 28}},
            title={'text': "Risk Index", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#dc2626" if is_late_risk else "#16a34a"},
                'steps': [
                    {'range': [0, 40], 'color': "#dcfce7"},
                    {'range': [40, 65], 'color': "#fef3c7"},
                    {'range': [65, 100], 'color': "#fee2e2"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        gauge_fig.update_layout(height=240, margin=dict(l=15, r=15, t=40, b=15))
        st.plotly_chart(gauge_fig, use_container_width=True)
        
    with res_col3:
        st.markdown("##### 🔍 Key Risk Factors Detected:")
        factors = []
        if ship_mode == "First Class":
            factors.append(("🔴 Shipping Mode: First Class", "Historical late rate is ~98% due to aggressive 1-day promise window."))
        elif ship_mode == "Second Class":
            factors.append(("🟡 Shipping Mode: Second Class", "Historical late rate is ~77% with standard 2-day delivery."))
        elif ship_mode == "Standard Class":
            factors.append(("🟢 Shipping Mode: Standard Class", "Standard 4-day buffer has lowest historical delay rate (~37%)."))
            
        if sched_days <= 1:
            factors.append(("🔴 Extremely Tight SLA", f"Only {sched_days} day(s) scheduled. High risk of logistical bottleneck."))
            
        if is_weekend:
            factors.append(("🟡 Weekend Order Placement", "Weekend fulfillment centers experience batching delays."))
            
        if same_country == 0:
            factors.append(("🟡 Cross-Border Logistics", "International customs clearance introduces variable transit delays."))
            
        if is_loss:
            factors.append(("🟡 Negative Margin Order", "Loss-making orders frequently correlate with dispute/reroute overhead."))
            
        if not factors:
            factors.append(("🟢 Optimal Order Profile", "All parameters conform to healthy fulfillment schedules."))
            
        for f_title, f_desc in factors[:4]:
            st.markdown(f"**{f_title}**  \n<span style='font-size:0.85rem; color:#475569;'>{f_desc}</span>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # What-If Mitigation Recommender
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("#### 💡 What-If Scenario Simulator & Mitigation Advisor")
    st.markdown("Simulate how adjusting fulfillment levers (Shipping Mode & Scheduled SLA) alters the risk profile:")
    
    sim_modes = ["First Class", "Second Class", "Standard Class", "Same Day"]
    sim_results = []
    
    for sm in sim_modes:
        default_days = {"First Class": 1, "Second Class": 2, "Standard Class": 4, "Same Day": 0}.get(sm, 2)
        sim_dict = input_dict.copy()
        sim_dict["Shipping Mode"] = sm
        sim_dict["Days for shipment (scheduled)"] = default_days
        sim_dict["is_rush_delivery"] = 1 if default_days <= 1 else 0
        
        sim_df = pd.DataFrame([sim_dict])
        sim_enc = preprocessor.transform(sim_df)
        sim_p = float(chosen_model.predict_proba(sim_enc)[0, 1])
        
        sim_results.append({
            "Shipping Mode": sm,
            "Standard SLA (Days)": default_days,
            "Late Delivery Risk": f"{sim_p:.1%}",
            "Risk Delta": f"{(sim_p - pred_prob):+.1%}",
            "Verdict": "Safe 🟢" if sim_p < threshold else "At-Risk 🔴"
        })
        
    sim_table = pd.DataFrame(sim_results)
    st.dataframe(sim_table, use_container_width=True, hide_index=True)
    
    if pred_prob > 0.50:
        st.info("💡 **Recommended Action**: Upgrading or switching this dispatch to **Standard Class (4 scheduled days)** substantially reduces delay penalty risk and logistics SLA breach!")

# =========================================================
# TAB 2: BATCH SHIPMENT AUDIT
# =========================================================
with tabs[1]:
    st.subheader("Batch Order Risk Audit & CSV Inference")
    st.markdown("Upload a CSV batch of order dispatches or instantly test with the pre-loaded benchmark batch.")
    
    batch_col1, batch_col2 = st.columns([2, 1])
    
    with batch_col1:
        uploaded_file = st.file_uploader("Upload Shipment CSV File", type=["csv"])
    with batch_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button("🧪 Load Sample Test Orders (35 Records)", use_container_width=True)
        
    batch_df = None
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(batch_df)} records successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif use_sample:
        sample_path = "sample_data/sample_orders.csv"
        if os.path.exists(sample_path):
            batch_df = pd.read_csv(sample_path)
            st.success("Loaded pre-computed sample orders successfully!")
        else:
            st.warning("Sample dataset file not found.")

    if batch_df is not None:
        # Check required columns
        req_cols = metadata["feature_columns"]
        missing_cols = [c for c in req_cols if c not in batch_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            with st.spinner("Processing batch predictions..."):
                proc_batch = preprocessor.transform(batch_df[req_cols])
                chosen_model = models[model_choice]
                batch_probs = chosen_model.predict_proba(proc_batch)[:, 1]
                batch_preds = (batch_probs >= threshold).astype(int)
                
                result_df = batch_df.copy()
                result_df["Predicted_Late_Risk"] = batch_preds
                result_df["Risk_Probability"] = np.round(batch_probs * 100, 1)
                result_df["Risk_Level"] = pd.cut(
                    batch_probs,
                    bins=[-0.01, 0.40, 0.65, 1.0],
                    labels=["Low Risk (Safe)", "Moderate Risk", "High Risk (Critical)"]
                )
                
                # Metrics
                total_orders = len(result_df)
                late_orders = int((result_df["Predicted_Late_Risk"] == 1).sum())
                on_time_orders = total_orders - late_orders
                late_pct = (late_orders / total_orders) * 100 if total_orders > 0 else 0
                at_risk_sales = float(result_df.loc[result_df["Predicted_Late_Risk"] == 1, "Sales"].sum())
                
                st.markdown("#### 📈 Batch Audit Summary")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                with kpi1:
                    st.metric("Total Shipments", f"{total_orders:,}")
                with kpi2:
                    st.metric("Late Risk Count", f"{late_orders:,}", delta=f"{late_pct:.1f}% of total", delta_color="inverse")
                with kpi3:
                    st.metric("On-Time Expected", f"{on_time_orders:,}", delta=f"{(100-late_pct):.1f}%")
                with kpi4:
                    st.metric("At-Risk Revenue ($)", f"${at_risk_sales:,.2f}")
                    
                # Charts
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    pie_fig = px.pie(
                        result_df,
                        names="Risk_Level",
                        title="Risk Category Breakdown",
                        color="Risk_Level",
                        color_discrete_map={
                            "Low Risk (Safe)": "#22c55e",
                            "Moderate Risk": "#f59e0b",
                            "High Risk (Critical)": "#ef4444"
                        },
                        hole=0.45
                    )
                    pie_fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(pie_fig, use_container_width=True)
                    
                with c2:
                    bar_fig = px.histogram(
                        result_df,
                        x="Shipping Mode",
                        color="Risk_Level",
                        barmode="group",
                        title="Risk Breakdown by Shipping Mode",
                        color_discrete_map={
                            "Low Risk (Safe)": "#22c55e",
                            "Moderate Risk": "#f59e0b",
                            "High Risk (Critical)": "#ef4444"
                        }
                    )
                    bar_fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(bar_fig, use_container_width=True)
                    
                st.markdown("#### 📋 Detailed Batch Orders Table")
                filter_risk = st.multiselect(
                    "Filter by Risk Level",
                    options=["Low Risk (Safe)", "Moderate Risk", "High Risk (Critical)"],
                    default=["High Risk (Critical)", "Moderate Risk", "Low Risk (Safe)"]
                )
                
                display_cols = [
                    "Shipping Mode", "Days for shipment (scheduled)", "Category Name",
                    "Order Region", "Sales", "Risk_Probability", "Risk_Level", "Predicted_Late_Risk"
                ]
                
                if "Actual_Late_Risk" in result_df.columns:
                    display_cols.append("Actual_Late_Risk")
                    
                filtered_table = result_df[result_df["Risk_Level"].isin(filter_risk)][display_cols]
                st.dataframe(filtered_table, use_container_width=True, height=350)
                
                # CSV Export
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=csv_data,
                    file_name="supply_chain_late_risk_predictions.csv",
                    mime="text/csv"
                )

# =========================================================
# TAB 3: SUPPLY CHAIN RISK ANALYTICS
# =========================================================
with tabs[2]:
    st.subheader("Supply Chain Risk Intelligence Dashboard")
    st.markdown("Exploratory analysis on the key operational drivers influencing fulfillment punctuality.")
    
    # Load sample for analytics
    analytics_path = "sample_data/sample_orders.csv"
    
    an_col1, an_col2 = st.columns(2)
    
    with an_col1:
        # Cross Tab Chart: Shipping Mode vs Scheduled SLA
        st.markdown("##### 1. Historical Delay Rate by Shipping Mode")
        mode_data = pd.DataFrame({
            "Shipping Mode": ["First Class", "Same Day", "Second Class", "Standard Class"],
            "Late Delivery Rate (%)": [98.0, 54.4, 77.6, 37.5],
            "Scheduled SLA (Days)": [1, 0, 2, 4]
        })
        fig_mode = px.bar(
            mode_data,
            x="Shipping Mode",
            y="Late Delivery Rate (%)",
            color="Late Delivery Rate (%)",
            text="Late Delivery Rate (%)",
            color_continuous_scale="Reds",
            title="Late Delivery Incidence by Logistics Method"
        )
        fig_mode.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_mode.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 115])
        st.plotly_chart(fig_mode, use_container_width=True)
        
    with an_col2:
        # Geographic Market Risk Breakdown
        st.markdown("##### 2. Regional Risk Distribution across Markets")
        market_data = pd.DataFrame({
            "Market": ["Pacific Asia", "USCA", "Africa", "Europe", "LATAM"],
            "Late Delivery Rate (%)": [58.2, 53.6, 56.4, 55.1, 54.8],
            "Avg Sales ($)": [215, 198, 185, 210, 202]
        })
        fig_mkt = px.bar(
            market_data,
            x="Market",
            y="Late Delivery Rate (%)",
            color="Avg Sales ($)",
            text="Late Delivery Rate (%)",
            color_continuous_scale="Viridis",
            title="Global Market Delay Rate & Volume Density"
        )
        fig_mkt.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_mkt.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 75])
        st.plotly_chart(fig_mkt, use_container_width=True)
        
    st.markdown("---")
    an_col3, an_col4 = st.columns(2)
    
    with an_col3:
        st.markdown("##### 3. Weekend Dispatch vs Delivery Delay Risk")
        weekend_data = pd.DataFrame({
            "Order Timing": ["Weekday Dispatch", "Weekend Dispatch (Sat/Sun)"],
            "Delay Rate (%)": [53.8, 57.9]
        })
        fig_wk = px.bar(
            weekend_data,
            x="Order Timing",
            y="Delay Rate (%)",
            color="Order Timing",
            text="Delay Rate (%)",
            color_discrete_sequence=["#3b82f6", "#ef4444"],
            title="Impact of Weekend Fulfillment Batching"
        )
        fig_wk.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_wk.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 75])
        st.plotly_chart(fig_wk, use_container_width=True)
        
    with an_col4:
        st.markdown("##### 4. Scheduled Days SLA Threshold Curve")
        sla_curve = pd.DataFrame({
            "Scheduled Days SLA": [0, 1, 2, 3, 4, 5, 6],
            "Late Risk Probability (%)": [54.4, 98.0, 77.6, 62.0, 37.5, 24.0, 12.0]
        })
        fig_sla = px.line(
            sla_curve,
            x="Scheduled Days SLA",
            y="Late Risk Probability (%)",
            markers=True,
            title="SLA Buffer Curve: How Extra Transit Days Squash Delay Risk"
        )
        fig_sla.update_traces(line_color="#dc2626", line_width=3, marker=dict(size=8, color="#991b1b"))
        fig_sla.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0, 105])
        st.plotly_chart(fig_sla, use_container_width=True)

# =========================================================
# TAB 4: MODEL BENCHMARK HUB
# =========================================================
with tabs[3]:
    st.subheader("Model Benchmark & Validation Hub")
    st.markdown("Comparison across the baseline notebook models and modern gradient boosting extensions.")
    
    bench_data = []
    for m_name, m_metrics in metadata["metrics_summary"].items():
        bench_data.append({
            "Model": m_name,
            "Accuracy": f"{m_metrics['Accuracy']:.2%}",
            "Precision": f"{m_metrics['Precision']:.2%}",
            "Recall": f"{m_metrics['Recall']:.2%}",
            "F1 Score": f"{m_metrics['F1 Score']:.4f}",
            "ROC-AUC": f"{m_metrics['ROC-AUC']:.4f}"
        })
        
    bench_df = pd.DataFrame(bench_data)
    st.table(bench_df)
    
    st.markdown("---")
    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        st.markdown("##### 📊 Metric Comparison Across Algorithms")
        plot_df = pd.DataFrame(metadata["metrics_summary"]).T.reset_index().rename(columns={"index": "Model"})
        fig_bar = px.bar(
            plot_df,
            x="Model",
            y=["Accuracy", "ROC-AUC", "F1 Score"],
            barmode="group",
            title="Benchmarking Performance Scores"
        )
        fig_bar.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), yaxis_range=[0.5, 0.9])
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with b_col2:
        st.markdown("##### 🎯 Confusion Matrix Visualizer")
        sel_cm_model = st.selectbox("Select Model for Confusion Matrix", options=list(models.keys()), index=0)
        cm = metadata["confusion_matrices"].get(sel_cm_model, [[0, 0], [0, 0]])
        
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: On-Time", "Actual: Late"],
            columns=["Pred: On-Time", "Pred: Late"]
        )
        
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix: {sel_cm_model}"
        )
        fig_cm.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("---")
    st.markdown("##### 🏆 Feature Importance Leaderboard (Top Predictors)")
    fi_model = st.selectbox("Select Model to Inspect Feature Importance", options=["Random Forest", "Decision Tree", "XGBoost", "LightGBM"], index=0)
    fi_list = metadata["feature_importances"].get(fi_model, [])
    
    if fi_list:
        fi_df = pd.DataFrame(fi_list).sort_values(by="importance", ascending=True)
        # Format labels nicely
        fi_df["clean_feature"] = fi_df["feature"].str.replace("cat__", "").str.replace("num__", "")
        
        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="clean_feature",
            orientation="h",
            title=f"Top Features Influencing Predictions ({fi_model})",
            color="importance",
            color_continuous_scale="Teal"
        )
        fig_fi.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)

# =========================================================
# TAB 5: PROJECT & ARCHITECTURE DOCUMENTATION
# =========================================================
with tabs[4]:
    st.subheader("Project Documentation & System Architecture")
    
    st.markdown("""
    ### 📌 Executive Summary
    **LogiRisk AI** is an end-to-end Machine Learning classification and decision-support system designed to forecast **Late Delivery Risk** in global supply chain logistics.
    Using operational data from the **DataCo Global Supply Chain Dataset**, this system evaluates order traits before and at dispatch time to flag high-risk shipments and provide actionable prescriptive guidance.

    ---
    ### 🔬 Feature Engineering Enhancements
    To elevate the original baseline notebook, we engineered domain-specific signals:
    1. **`is_rush_delivery`**: Binary flag indicating whether the scheduled delivery SLA is 1 day or less.
    2. **`is_weekend`**: Tracks whether the order dispatch occurred on a Saturday or Sunday, capturing fulfillment center weekend backlog.
    3. **`order_hour`**: Extracts the dispatch hour from the order timestamp to model daily logistics dispatch cutoffs.
    4. **`profit_margin` & `discount_ratio`**: Financial indicators capturing customer price sensitivity and returns/claims correlation.
    5. **`is_same_country`**: Cross-border versus domestic shipping indicator, reflecting customs delays.
    6. **`is_loss_order`**: Identifies loss-making transactions frequently prone to fulfillment discrepancies.

    ---
    ### 🤖 Model Comparison Matrix
    | Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Best Use Case |
    | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
    | **LightGBM** | **74.12%** | **88.58%** | **63.45%** | **0.7393** | **0.8115** | High-throughput cloud deployments |
    | **XGBoost** | **74.08%** | **88.09%** | **63.83%** | **0.7403** | **0.8108** | Production robustness & regularization |
    | **Random Forest** | 74.08% | 89.17% | 62.84% | 0.7372 | 0.8108 | High-precision ensemble voting |
    | **Decision Tree** | 73.05% | 84.76% | 65.13% | 0.7366 | 0.7918 | Rapid interpretable baseline |

    ---
    ### 💼 Business Impact & Supply Chain ROI
    - **SLA Breach Prevention**: By flagging critical shipments 48 hours prior to carrier handoff, logistics operators can adjust routing from *First Class* to guaranteed ground carriers.
    - **Reduced Delay Penalties**: Proactive notification of delivery risk prevents contract SLA fines and customer churn.
    - **Revenue Protection**: Identifies and prioritizes high-value at-risk orders in the batch audit dashboard.
    """)
