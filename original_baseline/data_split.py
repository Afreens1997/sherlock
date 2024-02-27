import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

# Load datasets
ds_train = load_dataset("swaghjal/sherlock-train")
ds_val = load_dataset("swaghjal/sherlock-val")

# Randomly select 10% of ds_train
num_samples_ds_train = int(0.1 * len(ds_train['train']))
subset_ds_train_indices = random.sample(range(len(ds_train['train'])), num_samples_ds_train)
subset_ds_train = [ds_train['train'][i] for i in subset_ds_train_indices]

# Split subset_ds_train into train and val with 80-20 split
train_data, val_data = train_test_split(subset_ds_train, test_size=0.2, random_state=42)

# Randomly select 30% of ds_val
num_samples_ds_val = int(0.3 * len(ds_val['val']))
subset_ds_val_indices = random.sample(range(len(ds_val['val'])), num_samples_ds_val)
subset_ds_val = [ds_val['val'][i] for i in subset_ds_val_indices]

# Convert datasets to JSON format
train_json = json.dumps(train_data)
val_json = json.dumps(val_data)
subset_val_json = json.dumps(subset_ds_val)

# Save datasets to JSON files
with open('../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_train.json', 'w') as f:
    f.write(train_json)

with open('../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_val.json', 'w') as f:
    f.write(val_json)

with open('../../../../data/tir/projects/tir4/users/svarna/Sherlock/data/subset_test.json', 'w') as f:
    f.write(subset_val_json)

# Print lengths of resulting datasets
print("Length of train data:", len(train_data))
print("Length of val data:", len(val_data))
print("Length of test:", len(subset_ds_val))
