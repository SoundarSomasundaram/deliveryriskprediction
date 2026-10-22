"""
Supply Chain Late Delivery Risk Classifier - Model Training Pipeline
Enhanced with advanced feature engineering, multi-model benchmarking (Decision Tree,
Random Forest, XGBoost, LightGBM), and artifact persistence for the Streamlit UI.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def build_features(df):
    """
    Apply domain-specific feature engineering to supply chain order records.
    """
    data = df.copy()
    
    # Parse dates
    order_date = pd.to_datetime(data['order date (DateOrders)'], errors='coerce')
    data['order_year'] = order_date.dt.year.fillna(2017).astype(int)
    data['order_month'] = order_date.dt.month.fillna(6).astype(int)
    data['order_day'] = order_date.dt.day.fillna(15).astype(int)
    data['order_dayofweek'] = order_date.dt.dayofweek.fillna(2).astype(int)
    data['order_hour'] = order_date.dt.hour.fillna(12).astype(int)
    
    # Domain features
    data['is_weekend'] = (data['order_dayofweek'] >= 5).astype(int)
    data['is_rush_delivery'] = (data['Days for shipment (scheduled)'] <= 1).astype(int)
    
    # Cross-border flag
    cust_country = data['Customer Country'].fillna('Unknown')
    order_country = data['Order Country'].fillna('Unknown')
    data['is_same_country'] = (cust_country == order_country).astype(int)
    
    # Financial indicators
    sales = data['Sales'].abs() + 1.0
    profit = data['Order Profit Per Order'].fillna(0.0)
    discount = data['Order Item Discount'].fillna(0.0)
    
    data['profit_margin'] = (profit / sales).round(4)
    data['discount_ratio'] = (discount / sales).round(4)
    data['is_loss_order'] = (profit < 0).astype(int)
    
    return data

def train_and_export():
    os.makedirs('models', exist_ok=True)
    os.makedirs('sample_data', exist_ok=True)
    
    data_path = 'data/DataCoSupplyChainDataset.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    print(f"Loading dataset from {data_path}...")
    # Read 50,000 representative records for fast, balanced, high-fidelity model training
    df_raw = pd.read_csv(data_path, encoding='latin-1', nrows=60000)
    print(f"Loaded {len(df_raw)} records.")
    
    # Target
    y = df_raw['Late_delivery_risk'].astype(int)
    
    # Feature Engineering
    df_feat = build_features(df_raw)
    
    feature_cols = [
        'Shipping Mode',
        'Days for shipment (scheduled)',
        'Type',
        'Customer Segment',
        'Department Name',
        'Category Name',
        'Market',
        'Order Region',
        'Order Item Quantity',
        'Product Price',
        'Sales',
        'Order Item Discount',
        'Order Profit Per Order',
        'order_month',
        'order_dayofweek',
        'order_hour',
        'is_weekend',
        'is_rush_delivery',
        'is_same_country',
        'profit_margin',
        'discount_ratio',
        'is_loss_order'
    ]
    
    X = df_feat[feature_cols].copy()
    
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"Categorical features ({len(cat_features)}): {cat_features}")
    print(f"Numerical features ({len(num_features)}): {num_features}")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            ('num', 'passthrough', num_features)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("\nFitting preprocessor...")
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    
    encoded_feature_names = preprocessor.get_feature_names_out().tolist()
    
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=120,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    }
    
    metrics_summary = {}
    confusion_matrices = {}
    feature_importances = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_encoded, y_train)
        
        preds = model.predict(X_test_encoded)
        probs = model.predict_proba(X_test_encoded)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        cm = confusion_matrix(y_test, preds).tolist()
        confusion_matrices[name] = cm
        
        metrics_summary[name] = {
            'Accuracy': round(float(acc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1 Score': round(float(f1), 4),
            'ROC-AUC': round(float(auc), 4)
        }
        
        print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Save model
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.joblib')
        
        # Extract feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_.tolist()
            fi_df = pd.DataFrame({'feature': encoded_feature_names, 'importance': importances})
            fi_df = fi_df.sort_values(by='importance', ascending=False).head(20)
            feature_importances[name] = fi_df.to_dict(orient='records')
            
    # Save preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Extract unique values for UI dropdowns
    categories_metadata = {}
    for col in cat_features:
        categories_metadata[col] = sorted(df_feat[col].dropna().unique().tolist())
        
    metadata = {
        'feature_columns': feature_cols,
        'cat_features': cat_features,
        'num_features': num_features,
        'metrics_summary': metrics_summary,
        'confusion_matrices': confusion_matrices,
        'feature_importances': feature_importances,
        'categories_metadata': categories_metadata,
        'dataset_size_used': len(df_raw),
        'test_samples_count': len(y_test)
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("\nSaved models and metadata to models/ directory successfully!")
    
    # Generate sample test orders CSV for UI batch prediction
    sample_df = df_feat[feature_cols].copy()
    sample_df['Actual_Late_Risk'] = y
    sample_orders = sample_df.sample(n=35, random_state=42)
    sample_orders.to_csv('sample_data/sample_orders.csv', index=False)
    print("Saved sample_orders.csv to sample_data/.")

if __name__ == '__main__':
    train_and_export()
