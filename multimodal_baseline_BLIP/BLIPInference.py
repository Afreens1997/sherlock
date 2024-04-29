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

PEFT_MODEL_ID = "svarna/blip_exp"
CONFIG = PeftConfig.from_pretrained(PEFT_MODEL_ID)

MODEL = Blip2ForConditionalGeneration.from_pretrained(CONFIG.base_model_name_or_path, load_in_8bit=True, device_map="auto")
MODEL = PeftModel.from_pretrained(MODEL, PEFT_MODEL_ID)

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
    parser.add_argument('test')
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

    with open(args.test) as f:
        test_file = json.load(f)

    inferences = {} 
    scorer = BERTScorer(model_type='bert-base-uncased')

    for data in test_file:
        image = Image.open(url2filepath(args.vg_dir, args.vcr_dir, data['inputs']['image']['url']))
        image = image.convert('RGBA')
        overlay = Image.new('RGBA', image.size, '#00000000')
        draw = ImageDraw.Draw(overlay, 'RGBA')
        for bbox in data['inputs']['bboxes']:
            left, top = bbox['left'], bbox['top']
            right, bottom = left + bbox['width'], top + bbox['height']
            # Draw the rectangle
            draw.rectangle(((left, top), (right, bottom)), fill='#ff05cd3c', outline='#05ff37ff', width=3)
        image = Image.alpha_composite(image, overlay)
        prompt = "Question: What objects can be seen in this image and what can you infer? Answer:"
        inputs = args.processor(images=image, text=prompt, return_tensors="pt").to(args.device, torch.float16)

        generated_ids = args.model.generate(**inputs, max_length=25)
        generated_caption = args.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        _, _, F1 = scorer.score([generated_caption], [data['targets']['inference']])
        F1 = F1.item()
        inferences[data['instance_id']] = F1
    
    return inferences



def main():
    args = parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inferences = inference(args)

    inferences = dict(sorted(inferences.items(), key=lambda x: x[1]))

    with open(args.result, "w") as f:
        json.dump(inferences, f)

if __name__ == '__main__':
    main()