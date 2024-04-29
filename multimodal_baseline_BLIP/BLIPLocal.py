import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

# PEFT_MODEL_ID = "svarna/blip_exp"
# CONFIG = PeftConfig.from_pretrained(PEFT_MODEL_ID)

MODEL = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)

PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

def url2filepath(vg_dir, vcr_dir, url):
    if 'VG_' in url:
        return vg_dir + '/'.join(url.split('/')[-2:])
    else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
        if 'vcr1images' in vcr_dir:
            return vcr_dir + '/'.join(url.split('/')[-2:])
        else:
            return vcr_dir + '/'.join(url.split('/')[-3:])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('result')
    parser.add_argument(
        '--vcr_dir',
        default='images/',
        help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')
    parser.add_argument(
        '--vg_dir',
        default='images/',
        help='directory with visual genome data, contains VG_100K and VG_100K_2')
    
    args = parser.parse_args()
    if args.vcr_dir[-1] != '/':
        args.vcr_dir += '/'
    if args.vg_dir[-1] != '/':
        args.vg_dir += '/'
    
    args.model = MODEL
    args.processor = PROCESSOR

    return args

def inference(args):

    with open(args.data) as f:
        data_file = json.load(f)

    inferences = {}

    for data in data_file:
        image = Image.open(url2filepath(args.vg_dir, args.vcr_dir, data['inputs']['image']['url']))
        local = {}
        for bbox in data['inputs']['bboxes']:
            left, top = bbox['left'], bbox['top']
            right, bottom = left + bbox['width'], top + bbox['height']
            img = image.crop((left,top,right,bottom))
            try:
                inputs = args.processor(images=img, return_tensors="pt").to(args.device, torch.float16)

                generated_ids = args.model.generate(**inputs, max_length=25)
                generated_caption = args.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                local[data['inputs']['clue']] = local.get(data['inputs']['clue'], []) + [generated_caption]
            except:
                print("Error with: ", data['instance_id'])
        print(local)
        data['local_captions'] = local
    
    return data_file



def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inferences = inference(args)

    with open(args.result, "w") as f:
        json.dump(inferences, f)

if __name__ == '__main__':
    main()