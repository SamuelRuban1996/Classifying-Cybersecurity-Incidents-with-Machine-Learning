
# Classifying Cybersecurity Incidents with Machine Learning

## Project Overview
This project aims to classify cybersecurity incidents based on historical data using a machine learning model. The model is designed to predict the incident grade, enabling organizations to prioritize and respond effectively to potential security threats. The pipeline includes steps such as data preprocessing, label encoding, feature scaling, and dimensionality reduction. The project involves training, evaluation, and deployment of a machine learning model to make predictions on new, unseen data.

## Key Features
- **Incident Classification:** Classifies cybersecurity incidents into various grades.
- **Data Preprocessing:** Handles missing values, unseen labels, and scales features.
- **Dimensionality Reduction:** Uses Truncated SVD or PCA for feature reduction.
- **Model Evaluation:** Evaluates model performance using F1-score, precision, and recall.
- **Deployment-Ready:** Includes saved model and preprocessing objects for easy deployment.

## Repository Structure
- `best_model.pkl`: Trained model saved using `joblib`.
- `preprocessing_objects.pkl`: Preprocessing objects including label encoder, target encoder, scaler, and dimensionality reduction model.
- `preprocessed_test_data.pkl`: Preprocessed test data used for predictions.
- `classify_incidents.py`: Main script for loading the model, making predictions, and evaluating the test data.
- `requirements.txt`: List of required packages to run the project.
- `README.md`: Project description, installation instructions, and usage guide.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/classifying-cybersecurity-incidents-ml.git
   cd classifying-cybersecurity-incidents-ml
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
After installing the dependencies, you can run the following script to evaluate the model and make predictions on the test data.

```bash
python classify_incidents.py
```

This script:
- Loads the saved model and preprocessing objects.
- Preprocesses the test data and handles unseen labels.
- Evaluates the model performance using F1-score, precision, and recall.
- Outputs the predictions on the test data.

## Model Evaluation
The model is evaluated using:
- **Macro F1-Score**: Measures the balance between precision and recall across multiple classes.
- **Precision**: The ability of the model to avoid false positives.
- **Recall**: The ability of the model to capture all relevant true positives.

## Requirements
See `requirements.txt` for the full list of dependencies.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
