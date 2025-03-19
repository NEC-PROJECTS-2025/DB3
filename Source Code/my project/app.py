import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables for preprocessing objects and model
numeric_imputer, non_numeric_imputer, scaler, encoder, selector, model = None, None, None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    global numeric_imputer, non_numeric_imputer, scaler, encoder, selector, model

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)  # Save file in the uploads folder

    # Load the uploaded CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400

    # Train model with uploaded data
    result = train_model(df)

    # Debugging: Print accuracy in terminal
    print(f"WRF Model Accuracy: {result.get('wrf_accuracy', 0)}%")

    # Pass result to result.html for displaying in a clean format
    return render_template('result.html', wrf_accuracy=f"{result.get('wrf_accuracy', 0)}%", report=result.get('classification_report', {}))


def train_model(df):
    global numeric_imputer, non_numeric_imputer, scaler, encoder, selector, model

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Handle Missing Values
    numeric_cols = X.select_dtypes(include=np.number).columns
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns

    numeric_imputer = SimpleImputer(strategy='mean')
    non_numeric_imputer = SimpleImputer(strategy='most_frequent')

    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    X[non_numeric_cols] = non_numeric_imputer.fit_transform(X[non_numeric_cols])

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(X[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(categorical_cols, axis=1)
        X = pd.concat([X, encoded_df], axis=1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Feature Selection
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train_resampled, y_train_resampled)
    X_train_selected = selector.transform(X_train_resampled)

    # Train Random Forest Model
    model = RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=10,
        min_samples_split=10, min_samples_leaf=5, class_weight='balanced'
    )
    model.fit(X_train_selected, y_train_resampled)

    # Save Model and Preprocessing Objects
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(selector, open('selector.pkl', 'wb'))
    pickle.dump(numeric_imputer, open('numeric_imputer.pkl', 'wb'))
    pickle.dump(non_numeric_imputer, open('non_numeric_imputer.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    pickle.dump(encoder, open('encoder.pkl', 'wb'))

    # Evaluate the model
    X_test_selected = selector.transform(X_test)
    predictions = model.predict(X_test_selected)

    wrf_accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions, output_dict=True)

    return {
        'wrf_accuracy': round(wrf_accuracy * 100, 2),  # Convert to percentage
        'classification_report': class_report
    }


if __name__ == '__main__':
    app.run(debug=True)
