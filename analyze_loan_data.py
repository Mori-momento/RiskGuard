# analyze_loan_data.py
import pandas as pd
import numpy as np

# Configure pandas to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Starting Data Analysis ---")

# Load the dataset
try:
    # Attempt to read with default comma delimiter
    try:
        df = pd.read_csv('Loan_Default.csv')
    except pd.errors.ParserError:
        print("ParserError encountered. Trying with skipping bad lines.")
        # If parsing fails (e.g., unexpected number of fields), try skipping problematic lines
        df = pd.read_csv('Loan_Default.csv', error_bad_lines=False, warn_bad_lines=True)

    print(f"\nDataset loaded successfully.")

    # Display basic information
    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Data Types &amp; Non-Null Counts ---")
    # df.info() provides a concise summary of dtypes and non-null counts
    df.info(verbose=True, show_counts=True)


    print("\n--- Missing Values Summary ---")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_summary = pd.DataFrame({'Missing Count': missing_values, 'Missing Percent': missing_percent})
    # Filter to show only columns with missing values and sort by percentage
    missing_summary_filtered = missing_summary[missing_summary['Missing Count'] > 0].sort_values(by='Missing Percent', ascending=False)

    if not missing_summary_filtered.empty:
        print(missing_summary_filtered)
    else:
        print("No missing values found.")

    # Display first few rows again to correlate with info
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Unique Values in Object Columns (Sample) ---")
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns[:15]: # Limit to first 15 object columns for brevity
        print(f"\nColumn: {col}")
        try:
            unique_vals = df[col].unique()
            print(f"Unique values ({len(unique_vals)}): {unique_vals[:10]} {'...' if len(unique_vals) > 10 else ''}") # Show first 10 unique values
        except Exception as e:
            print(f"Could not get unique values for {col}: {e}")


except FileNotFoundError:
    print("\nError: Loan_Default.csv not found in the current directory.")
except Exception as e:
    print(f"\nAn error occurred during analysis: {e}")

print("\n--- Analysis Complete ---")