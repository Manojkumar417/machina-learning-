from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# 1. Initialize the Flask application
app = Flask(__name__)


# Save the scaler object fitted on the ptbdb dataset
scaler = joblib.load('ptbdb_scaler.joblib')
print("StandardScaler object for PTBDB dataset saved successfully as 'ptbdb_scaler.joblib'.")
# Define paths for the model and scaler
MODEL_PATH = 'ptbdb_logistic_regression_model.joblib'
SCALER_PATH = 'ptbdb_scaler.joblib'

# Check if model and scaler files exist
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
if not os.path.exists(SCALER_PATH):
    print(f"Error: Scaler file not found at {SCALER_PATH}")
    exit()

# 2. Load the pre-trained logistic regression model and StandardScaler
try:
    ptbdb_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# 3. Define an API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request body
        data = request.get_json(force=True)
        
        # Ensure data is a list of features
        if not isinstance(data, list):
            return jsonify({'error': 'Input data must be a list of features.'}), 400

        # Convert the incoming data into a NumPy array and reshape for the scaler
        # Assuming input is a single sample, hence reshape(-1, 187) if it's 1D list of 187 elements
        # or directly to 2D if it's a list of lists already representing multiple samples.
        # For a single sample, we expect a list of 187 floats.
        input_features = np.array(data).reshape(1, -1) # Reshape to 1 sample, n_features
        
        # Preprocess the input data using the loaded StandardScaler's transform() method
        scaled_features = scaler.transform(input_features)
        
        # Make a prediction using the loaded logistic regression model's predict() method
        prediction = ptbdb_model.predict(scaled_features)
        
        # Return the prediction as a JSON response
        # Convert numpy int64 to standard Python int for jsonify
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. Add the if __name__ == '__main__': block to run the Flask application
if __name__ == '__main__':
    # This is for local development. For production, use a production-ready WSGI server.
    app.run(debug=True, host='0.0.0.0', port=5000)
