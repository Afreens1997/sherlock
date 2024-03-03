from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM, default_data_collator
from torch.utils.data import Dataset, DataLoader
import evaluate
import nltk
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
torch.cuda.empty_cache()


class Sherlock_Img2Txt_Dataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # returns batch size of 1
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding.update({"labels": encoding["input_ids"]})
        return encoding

root = 'PATH TO HIGHLIGHTED IMAGES'

train_dataset = load_dataset("imagefolder", data_dir=root, split="train")
val_dataset = load_dataset("imagefolder", data_dir=root, split="validation")
test_dataset = load_dataset("imagefolder", data_dir=root, split="test")

processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

train_dataset = Sherlock_Img2Txt_Dataset(train_dataset, processor)
val_dataset = Sherlock_Img2Txt_Dataset(val_dataset, processor)

print("data size")
print(len(train_dataset),len(val_dataset))

metric = evaluate.load("rouge")
ignore_pad_token_for_loss = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)

    decoded_predictions, decoded_labels = postprocess_text(decoded_predictions,decoded_labels)

    result = metric.compute(predictions=decoded_predictions,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != processor.tokenizer.pad_token_id) for pred in predicted
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


output_dir = "vishwa27/GIT_inf_w_bbox_caption_ep5"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=5e-5,
    num_train_epochs=5,
    fp16=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=4,
    save_total_limit=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    remove_unused_columns=False,
    push_to_hub=True,
    label_names=["labels"],
    load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator
)

trainer.train()

trainer.push_to_hub()