import json
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Blip2ForConditionalGeneration
import torch
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

DATA_PATH = "../../../../data/tir/projects/tir4/users/svarna/Sherlock/data"
TEST_FILE_PATH = f"{DATA_PATH}/subset_test.json"
VG_DIR = f"{DATA_PATH}/VG/"
VCR_DIR = f"{DATA_PATH}/VCR/"

test_data = json.load(open(TEST_FILE_PATH))

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def url2filepath(url):
    if 'VG_' in url:
        return VG_DIR + '/'.join(url.split('/')[-2:])
    else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
        if 'vcr1images' in VCR_DIR:
            return VCR_DIR + '/'.join(url.split('/')[-2:])
        else:
            return VCR_DIR + '/'.join(url.split('/')[-3:])

inf = {}
scorer = BERTScorer(model_type='bert-base-uncased')

for i, data in enumerate(test_data):
    image = Image.open(url2filepath(data['inputs']['image']['url']))
    # draw = ImageDraw.Draw(image)
    # for bbox in data['inputs']['bboxes']:
    #     left, top = bbox['left'], bbox['top']
    #     right, bottom = left + bbox['width'], top + bbox['height']
    #     label = '' # data['inputs']['clue']  # Get the label if it exists
        
    #         # Draw the rectangle
    #     draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
        
    #         # Draw the label if it exists
    #     if label:
    #         draw.text((left, top), label, fill="red")
    # prompt = "Question: What inference can be drawn by looking at the content inside the red rectangle? Do not include 'red rectangle' in your answer. Answer:"
    # prompt = "Focus on the given image. Now focus on the region within the red rectangle. Generate a caption for the image by infering from the view in the red rectangle. The generated caption is: "
    # prompt = "Generate a caption for the given image by focusing on the red rectangle. Generated caption is:"
    prompt = "Question: What objects can be seen in this image and what can you infer? Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    _, _, F1 = scorer.score([generated_text], [data['targets']['inference']])
    F1 = F1.item()
    inf[data['instance_id']] = F1

sorted_dict = dict(sorted(inf.items(), key=lambda x: x[1]))
print(sorted_dict)
    
with open(f"{DATA_PATH}/multimodal_baselines/inferences/BLIP_zero_shot_scores.json", "w") as file:
    json.dump(sorted_dict, file)