import json
from data_utils import get_clue
from file_utils import RESULTS_FOLDER, save_json, subset_test
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

model_checkpoint = "/home/afreens/projects/sherlock/results/checkpoint-9500"  # Adjust this to the path of your checkpoint
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

def run_inference(input_texts):
    # Tokenize the input texts
    inputs = tokenizer(input_texts, return_tensors="pt", padding="max_length", truncation=True)
    
    # Generate output sequences
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=512,
        max_new_tokens=10
    )
    
    # Decode the output sequences to strings
    outputs = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in output_sequences]
    
    return outputs

val_data = json.load(open(subset_test))
input_texts = [get_clue(x) for x in val_data]  # Replace these with your actual input texts
outputs = []

for i in range(0, len(input_texts), 10):
    batch_input = input_texts[i:i+10]
    batch_output = run_inference(batch_input)
    outputs.append(batch_output)

    save_json(outputs, f"{RESULTS_FOLDER}/unimodal_baselines/text/inference/flanT5_finetuned_subset_val_{10}_tokens1.json")
save_json(outputs, f"{RESULTS_FOLDER}/unimodal_baselines/text/inference/flanT5_finetuned_subset_val_{10}_tokens1.json")

for input_text, output in zip(input_texts, outputs):
    print(f"Input: {input_text}\nOutput: {output}\n")