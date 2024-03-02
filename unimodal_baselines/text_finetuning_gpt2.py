import json
import wandb
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
wandb.init(project="flant5_finetuning", entity="afreens")
from datasets import Dataset, DatasetDict
from data_utils import get_clue, get_inference
from file_utils import subset_train, subset_val
from transformers import Trainer, TrainingArguments

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


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize the input and output texts
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Resize the token embeddings since we added special tokens
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,   # Adjust based on your GPU memory
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()

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