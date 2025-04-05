# clean_loan_data.py
import pandas as pd
import numpy as np

print("--- Starting Data Cleaning ---")

# Load the dataset
try:
    df = pd.read_csv('Loan_Default.csv')
    print(f"Original dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- 1. Drop High-Missing Columns ---
    cols_to_drop_missing = ['Upfront_charges', 'Interest_rate_spread', 'rate_of_interest', 'dtir1']
    df.drop(columns=cols_to_drop_missing, inplace=True)
    print(f"Dropped high-missing columns: {cols_to_drop_missing}")

    # --- 2. Drop Identifier ---
    df.drop(columns=['ID'], inplace=True)
    print("Dropped ID column")

    # --- 3. Drop Constant Column (Year) ---
    if df['year'].nunique() == 1:
        df.drop(columns=['year'], inplace=True)
        print("Dropped constant 'year' column")
    else:
        print("'year' column has multiple values, keeping it.")


    # --- 4. Impute Missing Values ---
    # Numerical columns imputation (Median)
    num_cols_impute = ['property_value', 'LTV', 'income', 'term']
    for col in num_cols_impute:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Imputed missing values in '{col}' with median ({median_val})")

    # Categorical columns imputation (Mode)
    cat_cols_impute = ['loan_limit', 'approv_in_adv', 'loan_purpose', 'Neg_ammortization', 'age', 'submission_of_application']
    for col in cat_cols_impute:
         if col in df.columns:
            mode_val = df[col].mode()[0] # mode() returns a Series, take the first element
            df[col].fillna(mode_val, inplace=True)
            print(f"Imputed missing values in '{col}' with mode ('{mode_val}')")

    # --- 5. Convert Data Types &amp; Encode Categorical Features ---

    # Handle 'Gender' - replace 'Sex Not Available' with mode, then map
    gender_mode = df['Gender'].mode()[0]
    if gender_mode == 'Sex Not Available' and len(df['Gender'].mode()) > 1: # Find next common if mode is 'Sex Not Available'
         gender_mode = df['Gender'].mode()[1]
    df['Gender'].replace('Sex Not Available', gender_mode, inplace=True)
    print(f"Replaced 'Sex Not Available' in Gender with mode ('{gender_mode}')")
    # Example Binary Mapping (adjust based on actual unique values after imputation)
    # df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Joint': 2}).fillna(-1) # Or use One-Hot

    # Binary Mapping Function
    def map_binary(df, column, true_val, false_val=None):
        unique_vals = df[column].unique()
        if len(unique_vals) <= 2:
            # Automatically find the other value if only two exist
            if false_val is None:
                other_vals = [v for v in unique_vals if v != true_val]
                if len(other_vals) == 1:
                    false_val = other_vals[0]
                else: # Handle cases with only one value or unexpected scenarios
                     print(f"Warning: Could not automatically determine binary mapping for {column}. Skipping.")
                     return df
            print(f"Mapping '{column}': {true_val}=1, {false_val}=0")
            df[column] = df[column].map({true_val: 1, false_val: 0})
        else:
            print(f"Warning: Column '{column}' has more than 2 unique values ({unique_vals}). Skipping binary map, consider One-Hot.")
        return df

    df = map_binary(df, 'approv_in_adv', 'pre', 'nopre')
    df = map_binary(df, 'open_credit', 'opc', 'nopc')
    df = map_binary(df, 'business_or_commercial', 'b/c', 'nob/c')
    df = map_binary(df, 'Neg_ammortization', 'neg_amm', 'not_neg')
    df = map_binary(df, 'interest_only', 'int_only', 'not_int')
    df = map_binary(df, 'lump_sum_payment', 'lpsm', 'not_lpsm')
    df = map_binary(df, 'submission_of_application', 'to_inst', 'not_inst')
    df = map_binary(df, 'Secured_by', 'home', 'land') # Assuming 'home' and 'land' are the only options
    df = map_binary(df, 'Security_Type', 'direct', 'Indriect') # Check unique values first

    # Ordinal Mapping
    df['Credit_Worthiness'] = df['Credit_Worthiness'].map({'l1': 1, 'l2': 2})
    print("Mapped 'Credit_Worthiness' (l1:1, l2:2)")
    df['total_units'] = df['total_units'].map({'1U': 1, '2U': 2, '3U': 3, '4U': 4})
    print("Mapped 'total_units' (1U:1, 2U:2, 3U:3, 4U:4)")

    # Age Mapping (Midpoint)
    age_map = {
        '25-34': 29.5, '35-44': 39.5, '45-54': 49.5,
        '55-64': 59.5, '65-74': 69.5, '>74': 79.5, # Assuming 79.5 for >74
        '<25': 20.0 # Assuming 20 for <25
    }
    df['age'] = df['age'].map(age_map)
    print("Mapped 'age' ranges to midpoints")

    # One-Hot Encoding for remaining nominal categoricals
    cols_to_one_hot = [
        'loan_limit', 'Gender', 'loan_type', 'loan_purpose',
        'construction_type', 'occupancy_type', 'credit_type',
        'co-applicant_credit_type', 'Region'
    ]
    # Filter out columns that might have been dropped or already converted
    cols_to_one_hot = [col for col in cols_to_one_hot if col in df.columns and df[col].dtype == 'object']

    if cols_to_one_hot:
        print(f"Applying One-Hot Encoding to: {cols_to_one_hot}")
        df = pd.get_dummies(df, columns=cols_to_one_hot, drop_first=True, dummy_na=False) # drop_first to avoid multicollinearity
    else:
        print("No columns require One-Hot Encoding.")


    # Final check for missing values
    print("\n--- Missing Values After Cleaning ---")
    missing_after = df.isnull().sum().sum()
    if missing_after == 0:
        print("No missing values remain.")
    else:
        print(f"Warning: {missing_after} missing values still remain.")
        print(df.isnull().sum()[df.isnull().sum() > 0])


    # --- 6. Save Cleaned Data ---
    output_file = 'Loan_Default_Cleaned.csv'
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to '{output_file}'. Shape: {df.shape}")


except FileNotFoundError:
    print("\nError: Loan_Default.csv not found.")
except KeyError as e:
    print(f"\nError: Column not found during processing - {e}. Check column names.")
except Exception as e:
    print(f"\nAn error occurred during cleaning: {e}")

print("\n--- Cleaning Complete ---")