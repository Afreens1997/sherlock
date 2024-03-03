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
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
from tqdm import tqdm
import json


class Sherlock_Img2Txt_Dataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]
        encoding = self.processor(images=item["image"],return_tensors="pt")
        # returns batch size of 1
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding.update({"text": item['text']})
        return encoding

root = '/data/tir/projects/tir7/user_data/vishwavs/mmml_2024/bboxed_images_highlighted/test'

test_dataset = load_dataset("imagefolder", data_dir=root, split="train")

processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("vishwa27/GIT_inf_w_bbox_caption_ep5")

test_dataset = Sherlock_Img2Txt_Dataset(test_dataset, processor)

metric = evaluate.load("rouge")
ignore_pad_token_for_loss = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)
model.eval()

test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)

progress_bar = tqdm(range(len(test_dataloader)))

predicted_inf = []
img_ids = []
true_inf = []
with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_dataloader)):
        true_vals = batch.pop("text")
        pixel_values = batch.pop("pixel_values").to(device)

        outputs = model.generate(pixel_values=pixel_values, max_length=100)

        # predicted = outputs.argmax(-1)
        decoded_predictions = processor.batch_decode(outputs, skip_special_tokens=True)
        # print(decoded_predictions)
        decoded_predictions = decoded_predictions

        # print(predicted_inf)
        # print(true_inf)

        predicted_inf.extend(decoded_predictions)
        true_inf.extend(true_vals)
        progress_bar.update(1)

with open('predictions/git-bbox_caption_w_inf_predictions.json','w') as f_out:
    for i in range(len(predicted_inf)):
        obj = {}
        obj['actual_inf'] = true_inf[i]
        obj['predicted'] = predicted_inf[i]
        f_out.write(json.dumps(obj) + "\n")

assert len(predicted_inf) == len(test_dataset)

# print(predictions)
# print(predictions.shape)
