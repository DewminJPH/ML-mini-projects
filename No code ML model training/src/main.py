import os

import streamlit as st
import streamlit_option_menu as option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml_utility import (read_data,
                        preprocess_data,
                        train_model,
                        evaluate_model)

# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# get the parent directory
parent_dir = os.path.dirname(working_dir)

st.set_page_config(page_title="Automate ML", 
                   page_icon="🧠",
                   layout="centered")

st.title("🤖⚙️ No Code ML Model Training")

dataset_list = os.listdir(f"{parent_dir}/datasets")

dataset = st.selectbox("Select a dataset from the dropdown", 
                       dataset_list, 
                       index=None)

df = read_data(dataset)

if df is not None:
    st.dataframe(df.head())

    col1, col2, col3, col4 = st.columns(4)

    scaler_type_list = ["Standard Scaler", "Min-Max Scaler"]
    
    model_dictionary ={
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBoost Classifier": XGBClassifier()
    }

    with col1:
        target_column = st.selectbox("Select the target column", list(df.columns), index=None)
    with col2:
        scaler_type = st.selectbox("Select the scaler type", scaler_type_list, index= None)
    with col3:
        model_type = st.selectbox("Select the model type", list(model_dictionary.keys()), index=None)
    with col4:
        model_name = st.text_input("Model Name")
    
    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
        model_to_be_trained = model_dictionary[model_type]
        model = train_model(X_train, y_train, model_to_be_trained , model_name)
        accuracy = evaluate_model(model, X_test, y_test)

        st.success("Test Accuracy: " + str(accuracy))