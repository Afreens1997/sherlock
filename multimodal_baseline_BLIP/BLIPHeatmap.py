import json
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Blip2ForConditionalGeneration
import torchvision.transforms as transforms
import torch
from peft import PeftModel, PeftConfig
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer
import cv2
from tqdm import tqdm

DATA_PATH = "../../../../data/tir/projects/tir4/users/svarna/Sherlock/data"
TEST_FILE_PATH = f"{DATA_PATH}/subset_test.json"
VG_DIR = f"{DATA_PATH}/VG/"
VCR_DIR = f"{DATA_PATH}/VCR/"

test_data = json.load(open(TEST_FILE_PATH))


# PEFT_MODEL_ID = "svarna/blip_exp_no_bbox"
# CONFIG = PeftConfig.from_pretrained(PEFT_MODEL_ID)

# MODEL = Blip2ForConditionalGeneration.from_pretrained(CONFIG.base_model_name_or_path, load_in_8bit=True, device_map="auto")
# MODEL = PeftModel.from_pretrained(MODEL, PEFT_MODEL_ID)

MODEL = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

def url2filepath(url):
    if 'VG_' in url:
        return VG_DIR + '/'.join(url.split('/')[-2:])
    else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
        if 'vcr1images' in VCR_DIR:
            return VCR_DIR + '/'.join(url.split('/')[-2:])
        else:
            return VCR_DIR + '/'.join(url.split('/')[-3:])

def preprocess_image(image_path):
    processor = PROCESSOR
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    return pixel_values

def generate_caption(pixel_values):
    model = MODEL
    batch_size = pixel_values.shape[0]
    dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long)
    outputs = model(pixel_values=pixel_values, input_ids=dummy_input_ids, output_attentions=True)
    # print([logits.squeeze() for logits in outputs.logits])
    # captions = [PROCESSOR.decode(logits.squeeze(), skip_special_tokens=True) for logits in outputs.logits]
    # print(captions)
    attention_weights = outputs.vision_outputs.attentions[-1].squeeze(1)
    return attention_weights

def generate_heatmap(image_path, data):
    pixel_values = preprocess_image(image_path)
    attention_weights = generate_caption(pixel_values)
    img = Image.open(image_path)
    att_mat = torch.mean(attention_weights, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    min_val = aug_att_mat.min()
    max_val = aug_att_mat.max()
    aug_att_mat = (aug_att_mat - min_val) / (max_val - min_val)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis].squeeze(2)
    # print(mask.shape)
    # print(img.size)
    # result = (mask * img).astype("uint8")
    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    # ax1.set_title('Original')
    # ax2.set_title('Attention Map')
    # _ = ax1.imshow(img)
    # _ = ax2.imshow(result)
    # plt.savefig("attention.png")

    total_sum = 0
    for bbox in data['inputs']['bboxes']:
        left, top = bbox['left'], bbox['top']
        right, bottom = left + bbox['width'], top + bbox['height']
        for i in range(left, right):
            for j in range(top, bottom):
                if j < mask.shape[0] and i < mask.shape[1]:
                    total_sum += mask[j][i]
                else:
                    break
        total_sum /= (bbox['width'] * bbox['height'])
    return total_sum


# Example usage
ts = 0
for i, data in enumerate(test_data):
    image_path = url2filepath(data['inputs']['image']['url'])
    ts += generate_heatmap(image_path, data)

print(ts)
