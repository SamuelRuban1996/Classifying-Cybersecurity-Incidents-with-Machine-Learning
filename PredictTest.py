# Importing required libraries for loading models, handling data, and making predictions
import joblib  # For loading saved models and preprocessing objects
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score  # For model evaluation
from sklearn.impute import SimpleImputer  # For handling missing values

# Step 1: Load saved model, encoders, and preprocessing objects
def load_saved_objects():
    """
    Loads the saved model, encoders, and preprocessing objects from disk.
    
    This function retrieves:
    - 'best_model.pkl': The trained classification model (e.g., RandomForest, XGBoost, etc.).
    - 'preprocessing_objects.pkl': A tuple containing:
        - label_encoder: Used to encode the target variable (e.g., IncidentGrade).
        - target_encoder: Used for encoding categorical features (e.g., MitreTechniques, OSVersion).
        - ipca: Dimensionality reduction model (e.g., TruncatedSVD or PCA).
        - scaler: StandardScaler used to normalize or scale the feature data.
        - X_columns: List of the feature columns used during training.
    - 'preprocessed_test_data.pkl': Preprocessed test data used for final predictions.
    
    Returns:
    - model: trained model.
    - label_encoder: LabelEncoder, used for encoding the target variable.
    - target_encoder: TargetEncoder, used for encoding categorical features.
    - ipca: TruncatedSVD or PCA object, used for dimensionality reduction.
    - scaler: StandardScaler, used for feature scaling.
    - X_columns: list, feature columns used during training.
    - test_data: DataFrame, preprocessed test data.
    """
    model = joblib.load('best_model.pkl')  # Load the trained model
    label_encoder, target_encoder, ipca, scaler, X_columns = joblib.load('preprocessing_objects.pkl')  # Load preprocessing objects
    test_data = joblib.load('preprocessed_test_data.pkl')  # Load preprocessed test data
    return model, label_encoder, target_encoder, ipca, scaler, X_columns, test_data

# Step 2: Handle unseen labels in test data
def handle_unseen_labels(y_test, label_encoder):
    """
    Handles unseen labels in the test data that were not present during training.
    
    Parameters:
    - y_test: Series, the test labels.
    - label_encoder: LabelEncoder, encoder used during training.
    
    Returns:
    - y_test_fixed: Series, test labels with unseen labels replaced by the most frequent label.
    """
    # Identify unseen labels (those not present in the training set)
    known_classes = set(label_encoder.classes_)
    unseen_labels = set(y_test) - known_classes

    # Replace unseen labels with the most frequent label in the training set
    most_frequent_label = label_encoder.classes_[np.argmax(np.bincount(label_encoder.transform(label_encoder.classes_)))]
    y_test_fixed = y_test.apply(lambda x: most_frequent_label if x in unseen_labels else x)

    return y_test_fixed

# Step 3: Manually scale features using the saved scaler
def manual_scale(X, scaler, feature_columns):
    """
    Manually scales the feature set using the saved scaler.
    
    Parameters:
    - X: DataFrame, the input feature set.
    - scaler: StandardScaler, used for scaling the features.
    - feature_columns: list, columns to be scaled.
    
    Returns:
    - X_scaled: DataFrame, scaled features.
    """
    # Select only the features that were used during training
    X_selected = X[feature_columns]
    
    # Apply scaling using the saved scaler parameters
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        # Ensure consistency between scaler and input features
        if len(scaler.mean_) != X_selected.shape[1]:
            raise ValueError(f"Mismatch between scaler mean ({len(scaler.mean_)}) and selected features ({X_selected.shape[1]})")
        X_scaled = (X_selected - scaler.mean_) / scaler.scale_
    elif hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
        # For MinMaxScaler, use min_ and scale_ attributes
        if len(scaler.min_) != X_selected.shape[1]:
            raise ValueError(f"Mismatch between scaler min ({len(scaler.min_)}) and selected features ({X_selected.shape[1]})")
        X_scaled = (X_selected - scaler.min_) / scaler.scale_
    else:
        raise ValueError("Unexpected scaler type. Unable to manually scale the data.")
    
    return X_scaled

# Step 4: Impute missing values
def impute_missing_values(X):
    """
    Imputes missing values in the input feature set using a specified strategy.
    
    Parameters:
    - X: DataFrame, input feature set.
    
    Returns:
    - X_imputed: array, feature set with missing values imputed.
    """
    imputer = SimpleImputer(strategy='mean')  # You can adjust this strategy as needed
    X_imputed = imputer.fit_transform(X)
    return X_imputed

# Step 5: Make predictions with the preprocessed test data
def make_predictions_with_test_data():
    """
    Loads saved objects, preprocesses the test data, and makes predictions on it.
    
    Returns:
    - test_data: DataFrame, test data with predicted labels.
    """
    # Load saved model and preprocessing objects
    model, label_encoder, target_encoder, ipca, scaler, X_columns, test_data = load_saved_objects()

    # Separate features and target from the test data
    X_test = test_data.drop('IncidentGrade', axis=1)
    y_test = test_data['IncidentGrade']

    print(f"Test data shape: {X_test.shape}")
    print(f"Expected features count: {len(X_columns)}")

    # Ensure the test data has the correct columns
    X_test = X_test.reindex(columns=X_columns, fill_value=0)
    
    # Select the correct columns for scaling
    feature_columns = X_columns[:4]  # Assuming the scaler was trained on the first 4 columns

    # Manually scale the test data
    X_test_scaled = manual_scale(X_test, scaler, feature_columns)
    
    # Impute any missing values in the scaled test data
    X_test_scaled_imputed = impute_missing_values(X_test_scaled)
    
    # Perform dimensionality reduction using the saved PCA/SVD model
    X_test_reduced = ipca.transform(X_test_scaled_imputed)

    # Handle any unseen labels in the test set
    y_test_fixed = handle_unseen_labels(y_test, label_encoder)

    # Make predictions using the trained model
    y_pred = model.predict(X_test_reduced)

    # Add predicted labels to the test data and drop the original labels
    test_data['PredictedIncidentGrade'] = label_encoder.inverse_transform(y_pred)
    test_data = test_data.drop(columns=['IncidentGrade'])
    
    return test_data

# Step 6: Evaluate the model on the test data
def evaluate_test_data():
    """
    Evaluates the model on the test data using F1 score, precision, and recall.
    
    Returns:
    - f1: float, macro F1-score of the model on the test set.
    - precision: float, macro precision score of the model.
    - recall: float, macro recall score of the model.
    """
    # Load saved model and preprocessing objects
    model, label_encoder, target_encoder, ipca, scaler, X_columns, test_data = load_saved_objects()

    # Separate features and target from the test data
    X_test = test_data.drop('IncidentGrade', axis=1)
    y_test = test_data['IncidentGrade']

    # Ensure the test data has the correct columns
    feature_columns = X_columns[:4]  # Assuming the first 4 columns were used for scaling
    X_test = X_test.reindex(columns=X_columns, fill_value=0)

    # Handle any unseen labels in the test set
    y_test_fixed = handle_unseen_labels(y_test, label_encoder)

    # Manually scale the test data using the correct columns
    X_test_scaled = manual_scale(X_test, scaler, feature_columns)
    
    # Impute any missing values in the scaled test data
    X_test_scaled_imputed = impute_missing_values(X_test_scaled)
    
    # Perform dimensionality reduction using the saved PCA/SVD model
    X_test_reduced = ipca.transform(X_test_scaled_imputed)
    
    # Make predictions using the trained model
    y_pred = model.predict(X_test_reduced)
    y_test_labels = label_encoder.transform(y_test_fixed)

    # Calculate evaluation metrics
    f1 = f1_score(y_test_labels, y_pred, average='macro')
    precision = precision_score(y_test_labels, y_pred, average='macro')
    recall = recall_score(y_test_labels, y_pred, average='macro')

    print(f"Macro F1-Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return f1, precision, recall

# Step 7: Main function to run the evaluation and make predictions
def main():
    """
    Main function to evaluate the model and make predictions on the test data.
    """
    # Evaluate the model on the test set
    evaluate_test_data()
    
    # Make predictions on the test set and display the results
    test_data_with_predictions = make_predictions_with_test_data()
    print("\nTest Data with Predictions:")
    print(test_data_with_predictions.head())

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
