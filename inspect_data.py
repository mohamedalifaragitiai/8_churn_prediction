import pandas as pd

# Make sure this path is correct
DATA_FILE = 'data/customer_churn_mini.json' 

try:
    df = pd.read_json(DATA_FILE, lines=True)
    
    # Filter for logged-in users to reduce noise
    df_logged_in = df[df['auth'] == 'Logged In'].copy()

    # Get all unique page events and print them
    unique_pages = sorted(df_logged_in['page'].unique())
    
    print("--- Unique Page Events Found in Your Data ---")
    for page in unique_pages:
        print(page)
    print("---------------------------------------------")

except FileNotFoundError:
    print(f"ERROR: The file was not found at '{DATA_FILE}'. Please ensure the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")