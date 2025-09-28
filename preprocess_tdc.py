from tdc.single_pred import Tox
import pandas as pd

# Select a label from the label list you printed
label_name = 'NR-AR'

# Load the Tox21 dataset for the chosen assay
data = Tox(name='Tox21', label_name=label_name)
df = data.get_data()

# Drop rows with missing labels and keep only binary values (0 or 1)
df = df.dropna()
df = df[df['Y'].isin([0, 1])]

# Rename columns to match expected format
df = df.rename(columns={'Drug': 'smiles', 'Y': 'label'})

# Save to CSV
df.to_csv('data/tox21_binary.csv', index=False)

print(f"✅ Saved data/tox21_binary.csv with {len(df)} samples for label '{label_name}'")
