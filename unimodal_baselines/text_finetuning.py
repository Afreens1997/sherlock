import json
import wandb
wandb.init(project="flant5_finetuning", entity="afreens")
from datasets import Dataset, DatasetDict
from data_utils import get_clue, get_inference
from file_utils import subset_train, subset_val
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load your dataset
train_data = json.load(open(subset_train))
val_data = json.load(open(subset_val))

# Process the data into the format suitable for the model
def process_data(example):
    instruction = "Generate abductive inference from the clue provided. "
    # Combine instruction with the clue
    example["input_text"] = f"{instruction} Clue: {get_clue(example)} Inference:"
    # The target is the inference
    example["target_text"] = get_inference(example)
    return example

processed_train_data = [process_data(item) for item in train_data]
processed_val_data = [process_data(item) for item in val_data]

# Convert the processed data into a Hugging Face dataset
train_dataset = Dataset.from_dict({"input_text": [item["input_text"] for item in processed_train_data], 
                             "target_text": [item["target_text"] for item in processed_train_data]})
val_dataset = Dataset.from_dict({"input_text": [item["input_text"] for item in processed_val_data], 
                             "target_text": [item["target_text"] for item in processed_val_data]})

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize the input and output texts
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.001,
    report_to="wandb"
)

wandb.config = {
  "learning_rate": 2e-5,
  "epochs": 10,
  "batch_size": 8
}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()
wandb.finish()

trainer.save_model("/data/tir/projects/tir4/users/svarna/Sherlock/data/models/finetuned_flant5_model")