# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Starting Model Training & Evaluation ---")

# Define the top 7 features to use
top_features = [
    'LTV',
    'property_value',
    'credit_type_EQUI',
    'income',
    'Credit_Score',
    'loan_amount',
    'age'
]

# Load the cleaned dataset
try:
    df_cleaned = pd.read_csv('Loan_Default_Cleaned.csv')
    print(f"Cleaned dataset loaded: {df_cleaned.shape}")

    # Ensure all selected features exist in the dataframe
    missing_features = [f for f in top_features if f not in df_cleaned.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the cleaned data: {missing_features}")

    # Define features (X) using only the top 7 and target (y)
    X = df_cleaned[top_features]
    y = df_cleaned['Status']

    print(f"Selected features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # Split data into training and testing sets (using same random_state for consistency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Split data into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows)")

    # Initialize and train the Logistic Regression model
    # Using class_weight='balanced' to handle potential imbalance in the target variable
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000) # Increased max_iter for convergence
    model.fit(X_train, y_train)
    print("Training complete.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Non-Default (0)', 'Default (1)'])
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Default (0)', 'Default (1)'], yticklabels=['Non-Default (0)', 'Default (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    conf_matrix_filename = 'confusion_matrix.png'
    plt.savefig(conf_matrix_filename)
    print(f"\nConfusion matrix plot saved to '{conf_matrix_filename}'")


    # Save the trained model
    model_filename = 'loan_default_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nTrained model saved to '{model_filename}'")

except FileNotFoundError:
    print("\nError: Loan_Default_Cleaned.csv not found. Please ensure the cleaning script ran successfully.")
except ValueError as ve:
    print(f"\nError: {ve}")
except Exception as e:
    print(f"\nAn error occurred during model training: {e}")

print("\n--- Model Training & Evaluation Complete ---")