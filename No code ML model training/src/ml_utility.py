import os

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(working_dir)

# 1. Read the dataset
def read_data(file_name):
    file_path = f"{parent_dir}/datasets/{file_name}"
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    
# 2. Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    x = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_cols = x.select_dtypes(include=['number']).columns
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns

    if len(numerical_cols) == 0:
        pass
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # data imputation for missing values in numerical columns
        num_imputer = SimpleImputer(strategy='mean')
        x_train[numerical_cols] = num_imputer.fit_transform(x_train[numerical_cols])
        x_test[numerical_cols] = num_imputer.transform(x_test[numerical_cols])

        if scaler_type == "Standard Scaler":
            scaler = StandardScaler()
        elif scaler_type == "Min-Max Scaler":
            scaler = MinMaxScaler()
        
        # data scaling for numerical columns
        x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])
        x_test[numerical_cols] = scaler.transform(x_test[numerical_cols])

    if len(categorical_cols) == 0:
        pass
    
    else:
        # handling the missing values by filling the most_frequent word
        cat_imputer = SimpleImputer(strategy = "most_frequent")
        x_train[categorical_cols] = cat_imputer.fit_transform(x_train[categorical_cols])
        x_test[categorical_cols] = cat_imputer.fit_transform(x_test[categorical_cols])

        # to turn the categories into numbers
        encoder = OneHotEncoder()
        x_train_encoded = encoder.fit_transform(x_train[categorical_cols])
        x_test_encoded = encoder.fit_transform(x_test[categorical_cols])
        x_train_encoded = pd.DataFrame(x_train_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        x_test_encoded = pd.DataFrame(x_test_encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))
        x_train = pd.concat([x_train.drop(columns=categorical_cols), x_train_encoded], axis=1)
        x_test = pd.concat([x_test.drop(columns = categorical_cols), x_test_encoded], axis =1)

    return x_train, x_test, y_train, y_test

# 3. Train the model
def train_model(x_train, y_train, model, model_name):
    if not model_name or str(model_name).strip() == "":
        raise ValueError("model_name must be provided and non-empty")

    # Ensure output directory exists
    save_dir = os.path.join(parent_dir, "trained_model")
    os.makedirs(save_dir, exist_ok=True)

    model.fit(x_train, y_train)
    save_path = os.path.join(save_dir, f"{model_name}.pkl")
    with open(save_path, "wb") as file:
        pickle.dump(model, file)
    return model

# 4. Evaluate the model
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    return accuracy