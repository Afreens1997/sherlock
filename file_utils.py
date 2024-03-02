import json


DATA_PATH = "/data/tir/projects/tir4/users/svarna/Sherlock/data"
VisualGenome_VAL_FILE_PATH = f"{DATA_PATH}/subset_val.json"
VisualGenome_TRAIN_FILE_PATH = f"{DATA_PATH}/train.json"

subset_train = f"{DATA_PATH}/subset_train.json"
subset_val = f"{DATA_PATH}/subset_val.json"
subset_test = f"{DATA_PATH}/subset_test.json"

RESULTS_FOLDER = f"{DATA_PATH}/results"


def load_json(filename, sort_by_id = False):
    assert filename.endswith("json"), "file provided to load_json does not end with .json extension. Please recheck!"
    data = json.load(open(filename))
    return data

def save_json(data, filename):
    assert filename.endswith("json"), "file provided to save_json does not end with .json extension. Please recheck!"
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, sort_keys=False)