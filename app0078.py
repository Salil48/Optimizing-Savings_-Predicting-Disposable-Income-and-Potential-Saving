import streamlit as st
import pandas as pd
import joblib
import os

# ⚡️ Load features and models efficiently
@st.cache_resource
def load_feature_columns():
    return joblib.load("feature_columns.pkl")

@st.cache_resource
def load_models():
    models = {}
    for target in targets:
        models[target] = joblib.load(f"model_{target}.pkl")
    return models

# 🎯 Prediction targets
targets = [
    'Disposable_Income', 'Desired_Savings', 'Potential_Savings_Groceries',
    'Potential_Savings_Transport', 'Potential_Savings_Eating_Out',
    'Potential_Savings_Entertainment', 'Potential_Savings_Utilities',
    'Potential_Savings_Healthcare', 'Potential_Savings_Education',
    'Potential_Savings_Miscellaneous'
]

topredict_features = load_feature_columns()
models = load_models()

# 🎨 Streamlit UI
st.title("💸 Fast Income & Savings Predictor")

# 🧾 User Inputs
user_input = {}
for feature in topredict_features:
    if "Occupation_" in feature or "City_Tier_" in feature:
        continue
    elif feature in ["Income", "Rent", "Loan_Repayment", "Insurance", "Age"]:
        user_input[feature] = st.number_input(f"{feature.replace('_', ' ')}", min_value=0.0, step=100.0)
    elif feature == "Dependents":
        user_input[feature] = st.number_input("Dependents", min_value=0, step=1)
    elif feature in ['Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']:
        user_input[feature] = st.number_input(f"Monthly {feature.replace('_', ' ')}", min_value=0.0, step=50.0)

# 🔠 Encodings
occupation = st.selectbox("Occupation", ["Professional", "Retired", "Self_Employed", "Student"])
for occ in ["Occupation_Professional", "Occupation_Retired", "Occupation_Self_Employed", "Occupation_Student"]:
    user_input[occ] = 1 if occ.split("_")[1] == occupation else 0

tier = st.selectbox("City Tier", ["Tier_1", "Tier_2", "Tier_3"])
for city in ["City_Tier_Tier_1", "City_Tier_Tier_2", "City_Tier_Tier_3"]:
    user_input[city] = 1 if city.split("_")[-1] == tier else 0

# 🧮 Derived Features
total_spending = sum(user_input.get(cat, 0.0) for cat in ['Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous'])
income = user_input.get("Income", 0.0) or 1

for cat in ['Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']:
    user_input[f"Pct_Income_{cat}"] = user_input[cat] / income

user_input["Total_Spending"] = total_spending
user_input["Pct_Income_Total_Spending"] = total_spending / income
user_input["Dependents_to_Income_Ratio"] = user_input["Dependents"] / income
user_input["High_Spender"] = int(user_input["Pct_Income_Total_Spending"] > 0.7)
user_input["Savings_Efficiency"] = 1 - user_input["Pct_Income_Total_Spending"]

# 📥 Prepare input DataFrame
input_df = pd.DataFrame([user_input])
missing_cols = set(topredict_features) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[topredict_features]

# 🔮 Predict
if st.button("Predict"):
    st.subheader("📊 Results")
    for target in targets:
        prediction = models[target].predict(input_df)[0]
        st.write(f"**{target.replace('_', ' ')}**: ₹{prediction:,.2f}")
