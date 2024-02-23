import json
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Blip2ForConditionalGeneration
import torch

DATA_PATH = "../../../../data/tir/projects/tir4/users/svarna/Sherlock/data"
VAL_FILE_PATH = f"{DATA_PATH}/sherlock_val_with_split_idxs_v1_1.json"

get_image_url = lambda x:x["inputs"]["image"]["url"]
get_image_dim = lambda x:{"width": x["inputs"]["image"]["width"], "height": x["inputs"]["image"]["height"]}
get_bboxes = lambda x:x["inputs"]["bboxes"]
get_clue = lambda x:x["inputs"]["clue"]
get_inference = lambda x:x["targets"]["inference"]
get_instance_id = lambda x:x["instance_id"]

val_data = json.load(open(VAL_FILE_PATH))
unique_urls = len(set([get_image_url(x) for x in val_data]))

urls_hash = {}

for data in val_data:
    urls_hash[get_image_url(data)] = urls_hash.get(get_image_url(data), []) + [data]

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def display_image_with_bboxes(image_url, bboxes):
    # Download the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Draw the bounding boxes on the image
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        # Calculate the bounding box coordinates
        left, top = bbox[0]['left'], bbox[0]['top']
        right, bottom = left + bbox[0]['width'], top + bbox[0]['height']
        label = bbox[0].get('label', '')  # Get the label if it exists
        
        # Draw the rectangle
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
        
        # Draw the label if it exists
        if label:
            draw.text((left, top), label, fill="red")
    
    return img

c = {}

for j, url in enumerate(urls_hash.keys()):
    bboxes = []
    captions = {}
    for i in range(len(urls_hash[url])):
        bboxes.append(get_bboxes(urls_hash[url][i]))
    original_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image_with_bbox = display_image_with_bboxes(url, bboxes) 
    
    original_inputs = processor(original_image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**original_inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    captions['original'] = generated_text

    # prompt = "Focus on the bounding boxes present in the image. By focusing on them the new caption for the image is as follows: "
    bbox_inputs = processor(image_with_bbox, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**bbox_inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    captions['with_bboxes'] = generated_text

    c[url] = captions

    if (j + 1) % 50 == 0:
        print(f'{j+1} images done')
        break

with open("captions.json", "w") as file:
    json.dump(c, file)