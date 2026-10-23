# Prepare Test set from super dataset using seed number 42

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load CSV and drop missing values
csv_path = r"C:\Users\ahmad\Documents\Programming\fyp\Dataset\super_dataset.csv"
df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

# 2. Split exactly as in train_early_predictor.py
seed = 42

train_df, temp_df = train_test_split(
    df, 
    test_size=0.30, 
    random_state=seed, 
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=(2/3), 
    random_state=seed, 
    stratify=temp_df["label"]
)

print(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# 3. Save test set to disk
output_path = "test_dataset.csv"
test_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Test dataset successfully saved to: {output_path}")
