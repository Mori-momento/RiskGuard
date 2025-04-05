# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'loan_default_model.joblib'
# Define the exact order of features the model expects
EXPECTED_FEATURES = [
    'LTV',
    'property_value',
    'credit_type_EQUI', # This was one-hot encoded, needs careful handling in the form
    'income',
    'Credit_Score',
    'loan_amount',
    'age'
]

# --- Load Model ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Routes ---
@app.route('/')
def home():
    """Renders the main input form page."""
    return render_template('index.html', features=EXPECTED_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    if model is None:
        return render_template('index.html', features=EXPECTED_FEATURES, error="Model not loaded. Cannot make predictions.")

    try:
        # Extract features from form, converting to float
        input_features = []
        form_errors = {}
        for feature in EXPECTED_FEATURES:
            value_str = request.form.get(feature)
            if value_str is None or value_str == '':
                 form_errors[feature] = "This field is required."
                 continue # Skip appending if empty

            try:
                # Special handling for the one-hot encoded feature
                if feature == 'credit_type_EQUI':
                     # Assuming the form sends '1' if Equifax, '0' otherwise
                     value = 1.0 if value_str == '1' else 0.0
                else:
                    value = float(value_str)

                input_features.append(value)
            except ValueError:
                form_errors[feature] = "Please enter a valid number."

        if form_errors:
             # Re-render form with errors if any field is invalid/missing
             return render_template('index.html', features=EXPECTED_FEATURES, errors=form_errors, form_values=request.form)

        # Ensure we have the correct number of features
        if len(input_features) != len(EXPECTED_FEATURES):
             raise ValueError(f"Incorrect number of features received. Expected {len(EXPECTED_FEATURES)}, got {len(input_features)}")

        # Convert to numpy array for the model
        final_features = np.array([input_features])

        # Make prediction and get probabilities
        prediction = model.predict(final_features)[0] # Get the single prediction
        probabilities = model.predict_proba(final_features)[0] # Get probabilities for the single prediction

        # Format results
        result_text = "Likely to Default" if prediction == 1 else "Unlikely to Default"
        probability_default = probabilities[1] * 100 # Probability of class 1 (Default)

        return render_template('index.html',
                               features=EXPECTED_FEATURES,
                               prediction_text=result_text,
                               prediction_prob=f"{probability_default:.2f}%",
                               form_values=request.form # Pass back form values to repopulate
                               )

    except ValueError as ve:
         # Handle potential errors during feature extraction or conversion
         return render_template('index.html', features=EXPECTED_FEATURES, error=f"Input Error: {ve}", form_values=request.form)
    except Exception as e:
        print(f"Prediction Error: {e}") # Log the error server-side
        return render_template('index.html', features=EXPECTED_FEATURES, error="An error occurred during prediction.", form_values=request.form)


# --- Run App ---
if __name__ == "__main__":
    # Use environment variable for port or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Set debug=True for development, allows auto-reloading
    app.run(debug=True, host='0.0.0.0', port=port)