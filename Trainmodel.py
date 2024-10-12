# Import necessary libraries
# Dask is used for scalable data manipulation, joblib for saving models, 
# and Dask-ML for machine learning tasks. We also import visualization 
# libraries and SMOTE for handling imbalanced data.

import pandas as pd
import dask.dataframe as dd
import dask.array as da
import joblib  # For saving and loading models
from dask_ml.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import category_encoders as ce
import numpy as np
import shap  # For model interpretability
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Step 1: Load and clean the data using Dask to handle large datasets
def load_and_clean_data(file_path, sample_size=None):
    """
    Loads data from a CSV file, cleans missing values, and samples if specified.
    
    Parameters:
    - file_path: str, path to the dataset (e.g., 'path/to/training_data.csv').
    - sample_size: int, number of rows to sample from the data (optional).
    
    Returns:
    - data_cleaned: Dask DataFrame, cleaned data.
    """
    # Define data types for columns to optimize memory usage
    dtypes = {
        'CountryCode': 'float64',
        'OSFamily': 'float64',
        'IncidentGrade': 'object',
        'OSVersion': 'float64',
        'Timestamp': 'object'
    }
    
    # Load the CSV file into a Dask DataFrame
    data = dd.read_csv(file_path, dtype=dtypes)

    # Keep only the required columns
    columns_to_keep = ['Timestamp', 'Category', 'IncidentGrade', 'MitreTechniques', 
                       'SuspicionLevel', 'OSFamily', 'OSVersion', 'CountryCode']
    existing_columns = [col for col in columns_to_keep if col in data.columns]
    data_cleaned = data[existing_columns]

    # Fill missing values with the mode or placeholders based on the column type
    for column in existing_columns:
        if column in ['Category', 'IncidentGrade']:
            mode = data_cleaned[column].dropna().mode().compute()
            if not mode.empty:
                data_cleaned[column] = data_cleaned[column].fillna(mode.iloc[0])
        elif column == 'Timestamp':
            data_cleaned[column] = data_cleaned[column].fillna(pd.NaT)
        else:
            data_cleaned[column] = data_cleaned[column].fillna('Unknown')

    # Optionally sample the data for faster processing
    if sample_size is not None:
        total_rows = data_cleaned.shape[0].compute()
        frac = sample_size / total_rows
        data_cleaned = data_cleaned.sample(frac=frac, random_state=42)

    return data_cleaned

# Step 2: Feature extraction from the timestamp and other columns
def feature_extraction(data):
    """
    Extracts new features such as hour and day of the week from the timestamp.
    
    Parameters:
    - data: Dask DataFrame, input data with a timestamp column.
    
    Returns:
    - data: Dask DataFrame, data with new extracted features.
    """
    # Convert the 'Timestamp' column to datetime
    data['Timestamp'] = dd.to_datetime(data['Timestamp'], errors='coerce')

    # Extract hour and day of the week from the 'Timestamp'
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek

    return data.drop(columns=['Timestamp'])

# Step 3: Encode categorical features using label and target encoding
def encode_features(data):
    """
    Encodes categorical features using label encoding and target encoding.
    
    Parameters:
    - data: Dask DataFrame, input data.
    
    Returns:
    - data: Dask DataFrame, encoded data.
    - label_encoder: LabelEncoder, encoder for the 'IncidentGrade' column.
    - target_encoder: TargetEncoder, encoder for categorical columns.
    """
    data['IncidentGrade'] = data['IncidentGrade'].fillna('Unknown')
    
    # Ensure columns are cast as strings for encoding
    columns_to_cast = ['Category', 'SuspicionLevel', 'OSFamily', 'CountryCode']
    for col in columns_to_cast:
        data[col] = data[col].astype(str)

    label_encoder = LabelEncoder()
    meta = ('IncidentGrade', int)
    data['IncidentGrade'] = data['IncidentGrade'].map_partitions(lambda x: label_encoder.fit_transform(x), meta=meta)

    # Apply get_dummies to categorical columns
    data = data.categorize(columns=columns_to_cast)
    data = dd.get_dummies(data, columns=columns_to_cast, drop_first=True)

    # Apply target encoding to complex features
    target_encoder = ce.TargetEncoder(cols=['MitreTechniques', 'OSVersion'])
    data = target_encoder.fit_transform(data.compute(), data['IncidentGrade'].compute())
    
    return dd.from_pandas(data, npartitions=5), label_encoder, target_encoder

# Step 4: Prepare the data by separating features and target
def prepare_data(data):
    """
    Separates the feature set (X) from the target variable (y).
    
    Parameters:
    - data: Dask DataFrame, input data with features and target.
    
    Returns:
    - X: Dask DataFrame, features.
    - y: Dask Series, target.
    """
    X = data.drop('IncidentGrade', axis=1)
    y = data['IncidentGrade']
    return X, y

# Step 5: Dimensionality reduction using Truncated SVD
def reduce_dimensionality(X, n_components=100):
    """
    Reduces the dimensionality of the feature set using Truncated SVD.
    
    Parameters:
    - X: Dask DataFrame, input feature set.
    - n_components: int, number of components for SVD.
    
    Returns:
    - X_reduced: array, reduced feature set.
    - svd: TruncatedSVD object, trained SVD model.
    - scaler: StandardScaler object, used for scaling the data.
    """
    X_numeric = X.select_dtypes(include=[np.number])
    X_array = X_numeric.to_dask_array(lengths=True)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_array.compute())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    n_components = min(n_components, X_scaled.shape[1])
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X_scaled)

    return X_reduced, svd, scaler

# Step 6: Balance the dataset using SMOTE
def balance_data_in_batches(X, y, batch_size=10000):
    """
    Balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique).
    
    Parameters:
    - X: array, feature set.
    - y: Dask Series, target variable.
    - batch_size: int, number of samples to process in each batch.
    
    Returns:
    - X_resampled_dd: Dask DataFrame, balanced feature set.
    - y_resampled_dd: Dask Series, balanced target variable.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y.compute())

    X_resampled_dd = dd.from_pandas(pd.DataFrame(X_resampled), npartitions=5)
    y_resampled_dd = dd.from_pandas(pd.Series(y_resampled), npartitions=5)

    return X_resampled_dd, y_resampled_dd

# Step 7: Split the dataset into training and testing sets
def split_data(X, y):
    """
    Splits the data into training and testing sets.
    
    Parameters:
    - X: Dask DataFrame, features.
    - y: Dask Series, target variable.
    
    Returns:
    - X_train, X_test, y_train, y_test: training and testing sets.
    """
    return dask_train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Step 8: Define the machine learning models to be evaluated
def get_models():
    """
    Returns a dictionary of different models to be trained.
    
    Returns:
    - models: dict, model names as keys and their instances as values.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(),
        'Neural Network': MLPClassifier(max_iter=1000)
    }
    return models

# Step 9: Train and tune the best model using RandomizedSearchCV
def train_best_model(X_train, y_train):
    """
    Trains the best model using RandomizedSearchCV on the training set.
    
    Parameters:
    - X_train: Dask DataFrame, training features.
    - y_train: Dask Series, training target.
    
    Returns:
    - best_estimator: trained model.
    """
    model = RandomForestClassifier()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
    search = RandomizedSearchCV(model, param_distributions=param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
    search.fit(X_train, y_train)
    return search.best_estimator_

# Step 10: Plot feature importance using SHAP for interpretable models
def plot_feature_importance(model, X_train):
    """
    Plots the feature importance using SHAP values for tree-based models.
    
    Parameters:
    - model: trained model, used for SHAP analysis.
    - X_train: Dask DataFrame, training features.
    """
    model_type = type(model).__name__
    X_train_pandas = X_train.compute()

    if model_type in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
        explainer = shap.TreeExplainer(model)
        X_train_pandas_sample = X_train_pandas.sample(n=10000, random_state=42) if X_train_pandas.shape[0] > 10000 else X_train_pandas
        shap_values = explainer.shap_values(X_train_pandas_sample)
        shap.summary_plot(shap_values, X_train_pandas_sample)
    else:
        print(f"SHAP not supported for {model_type}. Skipping feature importance for this model.")

# Step 11: Perform error analysis using confusion matrix and common misclassifications
def error_analysis(y_true, y_pred, label_encoder):
    """
    Analyzes model errors using confusion matrix and misclassification analysis.
    
    Parameters:
    - y_true: Dask Series, true labels.
    - y_pred: array, predicted labels.
    - label_encoder: LabelEncoder, used for decoding the target labels.
    """
    conf_matrix = confusion_matrix(y_true.compute(), y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for Error Analysis')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    misclassified = pd.DataFrame({'True': label_encoder.inverse_transform(y_true.compute()), 'Predicted': label_encoder.inverse_transform(y_pred)})
    print("\nError Analysis: Common Misclassifications")
    print(misclassified[misclassified['True'] != misclassified['Predicted']].head())

# Step 12: Save the trained model and preprocessed data for future use
def save_model_and_data(best_model, test_data, label_encoder, target_encoder, svd, scaler, X_columns):
    """
    Saves the trained model and preprocessing objects to disk for later use.
    
    Parameters:
    - best_model: trained model.
    - test_data: Dask DataFrame, preprocessed test data.
    - label_encoder: LabelEncoder, used for encoding the target variable.
    - target_encoder: TargetEncoder, used for encoding categorical features.
    - svd: TruncatedSVD, dimensionality reduction model.
    - scaler: StandardScaler, used for scaling the features.
    - X_columns: list, names of the feature columns.
    """
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump((label_encoder, target_encoder, svd, scaler, X_columns), 'preprocessing_objects.pkl')
    joblib.dump(test_data.compute(), 'preprocessed_test_data.pkl')

# Main function to run the entire process
def main(train_file_path, test_file_path):
    """
    Main function to load data, train model, and save results.
    
    Parameters:
    - train_file_path: str, path to the training data (e.g., 'path/to/train_data.csv').
    - test_file_path: str, path to the test data (e.g., 'path/to/test_data.csv').
    """
    # Step 1: Load and clean training data
    data = load_and_clean_data(train_file_path)
    
    # Step 2: Feature extraction
    data = feature_extraction(data)
    
    # Step 3: Encode features
    data, label_encoder, target_encoder = encode_features(data)
    
    # Step 4: Prepare features and target
    X, y = prepare_data(data)
    
    # Step 5: Dimensionality reduction
    X_reduced, svd, scaler = reduce_dimensionality(X)
    
    # Step 6: Balance the data
    X_balanced, y_balanced = balance_data_in_batches(X_reduced, y)
    
    # Step 7: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_balanced, y_balanced)
    
    # Step 8: Train the best model
    best_model = train_best_model(X_train, y_train)
    
    # Step 9: Analyze feature importance
    print("\nAnalyzing Feature Importance...")
    plot_feature_importance(best_model, X_train)

    # Step 10: Make predictions on the test set and analyze errors
    y_pred = best_model.predict(X_test)
    print("\nPerforming Error Analysis...")
    error_analysis(y_test, y_pred, label_encoder)

    # Step 11: Load and preprocess the test data
    test_data = load_and_clean_data(test_file_path)
    test_data = feature_extraction(test_data)
    test_data, _, _ = encode_features(test_data)

    # Step 12: Save the model and preprocessed data for later use
    save_model_and_data(best_model, test_data, label_encoder, target_encoder, svd, scaler, X.columns)

# Run the main function if this script is executed
if __name__ == "__main__":
    main('path/to/training_data.csv', 'path/to/test_data.csv')
