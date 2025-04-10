import pandas as pd
import glob

# Load all *_data.csv files
csv_files = glob.glob("*_data.csv")

# Combine them
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save as one combined file
df.to_csv("sign_data.csv", index=False)

print("Combined CSV saved as sign_data.csv with", len(df), "samples.")
