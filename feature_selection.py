# feature_selection.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Starting Feature Selection ---")

# Load the cleaned dataset
try:
    df_cleaned = pd.read_csv('Loan_Default_Cleaned.csv')
    print(f"Cleaned dataset loaded: {df_cleaned.shape}")

    # Define features (X) and target (y)
    X = df_cleaned.drop('Status', axis=1)
    y = df_cleaned['Status']

    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # Although we can calculate importance on the whole dataset,
    # splitting helps simulate a real modeling scenario.
    # We'll fit the model on the training data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Split data into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows)")

    # Initialize and train the RandomForestClassifier
    # n_estimators=100 is a common starting point
    # random_state for reproducibility
    # n_jobs=-1 to use all available CPU cores
    print("\nTraining RandomForestClassifier to determine feature importances...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced') # Added class_weight
    rf_model.fit(X_train, y_train)
    print("Training complete.")

    # Get feature importances
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the top N features
    top_n = 15
    print(f"\n--- Top {top_n} Most Important Features ---")
    print(feature_importance_df.head(top_n))

    # Optional: Plot feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
    plt.title(f'Top {top_n} Feature Importances from RandomForest')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plot_filename = 'feature_importances.png'
    plt.savefig(plot_filename)
    print(f"\nFeature importance plot saved to '{plot_filename}'")


except FileNotFoundError:
    print("\nError: Loan_Default_Cleaned.csv not found. Please ensure the cleaning script ran successfully.")
except Exception as e:
    print(f"\nAn error occurred during feature selection: {e}")

print("\n--- Feature Selection Complete ---")