import json
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Blip2ForConditionalGeneration
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter

PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
MODEL = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)

def url2filepath(vg_dir, vcr_dir, url):
    if 'VG_' in url:
        return vg_dir + '/'.join(url.split('/')[-2:])
    else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
        if 'vcr1images' in vcr_dir:
            return vcr_dir + '/'.join(url.split('/')[-2:])
        else:
            return vcr_dir + '/'.join(url.split('/')[-3:])

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, vg_dir, vcr_dir):
        self.dataset = dataset
        self.processor = processor
        self.vcr_dir = vcr_dir
        self.vg_dir = vg_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = Image.open(url2filepath(self.vg_dir, self.vcr_dir, item['inputs']['image']['url']))
        draw = ImageDraw.Draw(image)
        for bbox in item['inputs']['bboxes']:
            left, top = bbox['left'], bbox['top']
            right, bottom = left + bbox['width'], top + bbox['height']
            label = item['inputs']['clue']  # Get the label if it exists
        
            # Draw the rectangle
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
        
            # Draw the label if it exists
            if label:
                draw.text((left, top), label, fill="red")
        caption = item['targets']['inference']
        encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = caption
        return encoding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('val')

    parser.add_argument('--batch_size',
                        type=int,
                        default=256)

    parser.add_argument(
        '--vcr_dir',
        default='images/',
        help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')

    parser.add_argument(
        '--vg_dir',
        default='images/',
        help='directory with visual genome data, contains VG_100K and VG_100K_2')

    parser.add_argument('--lr',
                        type=float,
                        default=.00001)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=3)

    parser.add_argument('--output_dir',
                        type=str,
                        default='clip_model_outputs')

    parser.add_argument('--debug',
                        type=int,
                        default=0)

    parser.add_argument('--workers_dataloader',
                        type=int,
                        default=8)

    parser.add_argument('--save_every',
                        type=int,
                        help='if >1, a checkpoint will be saved every this many gradient updates.',
                        default=0)

    parser.add_argument('--early_stop',
                        type=int,
                        help='if > 0, if the loss doesnt improve in this many epochs, quit.',
                        default=5)

    parser.add_argument('--val_stat',
                        type=str,
                        help='which stat should we use for early stopping?',
                        default='loss',
                        choices=['loss', 'meanrank'])

    args = parser.parse_args()

    if args.vcr_dir[-1] != '/':
        args.vcr_dir += '/'
    if args.vg_dir[-1] != '/':
        args.vg_dir += '/'
    return args

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    processor = PROCESSOR
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

def main():
    args = parse_args()
    np.random.seed(1)

    model = MODEL
    processor = PROCESSOR

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"]
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    with open(args.train) as f:
        train_file = json.load(f)
    
    train_dataset = ImageCaptioningDataset(train_file, processor, args.vg_dir, args.vcr_dir)
    train = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)

    with open(args.val) as f:
        val_file = json.load(f)

    val_dataset = ImageCaptioningDataset(val_file, processor, args.vg_dir, args.vcr_dir)
    val = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        print("Epoch:", epoch)
        epoch_loss = 0
        model.train()
        for idx, batch in enumerate(train):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
    
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss /= len(train)
        print(f"Epoch {epoch} Loss: {epoch_loss}")

        model.eval()
        val_loss = 0
        for idx, batch in enumerate(val):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)

            loss = outputs.loss
            val_loss += loss.item()
        val_loss /= len(val)
        print(f"Epoch {epoch} Val Loss: {val_loss}")

    model.push_to_hub("svarna/blip_exp")

if __name__ == '__main__':
    main()