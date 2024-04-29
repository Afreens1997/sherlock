import json
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

DATA_PATH = "../../../../data/tir/projects/tir4/users/svarna/Sherlock/data"
SCORE_FILE_PATH_ZERO_SHOT = f"{DATA_PATH}/multimodal_baselines/inferences/BLIP_zero_shot_scores.json"
PREDICTION_FILE_PATH_ZERO_SHOT = f"{DATA_PATH}/multimodal_baselines/inferences/BLIP_zero_shot.json"
SCORE_FILE_PATH = f"{DATA_PATH}/multimodal_baselines/inferences/BLIP_scores.json"
PREDICTION_FILE_PATH = f"{DATA_PATH}/multimodal_baselines/inferences/BLIP.json"
SCORE_FILE_PATH_NOBBOX = f"{DATA_PATH}/multimodal_baselines/inferences/BLIP_no_bbox_scores.json"
PREDICTION_FILE_PATH_NOBBOX = f"{DATA_PATH}/multimodal_baselines/inferences/BLIP_no_bbox.json"
VG_DIR = f"{DATA_PATH}/VG/"
VCR_DIR = f"{DATA_PATH}/VCR/"
TEST_FILE_PATH = f"{DATA_PATH}/subset_test.json"

scores_data_zero_shot = json.load(open(SCORE_FILE_PATH_ZERO_SHOT))
prediction_data_zero_shot = json.load(open(PREDICTION_FILE_PATH_ZERO_SHOT))
scores_data = json.load(open(SCORE_FILE_PATH))
prediction_data = json.load(open(PREDICTION_FILE_PATH))
scores_data_nobbox = json.load(open(SCORE_FILE_PATH_NOBBOX))
prediction_data_nobbox = json.load(open(PREDICTION_FILE_PATH_NOBBOX))
test_data = json.load(open(TEST_FILE_PATH))

image_ids = list(scores_data_zero_shot.keys())
bottom_5 = image_ids[:10]
top_5 = image_ids[-10:]

def url2filepath(url):
    if 'VG_' in url:
        return VG_DIR + '/'.join(url.split('/')[-2:])
    else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
        if 'vcr1images' in VCR_DIR:
            return VCR_DIR + '/'.join(url.split('/')[-2:])
        else:
            return VCR_DIR + '/'.join(url.split('/')[-3:])

# for image in top_10:
#     prediction = prediction_data[image]
#     score = scores_data[image]
#     data = 
#     image = Image.open(url2filepath(data['inputs']['image']['url']))
#     print(prediction, score)
i = 0
for data in test_data:
    if data['instance_id'] in top_5 or data['instance_id'] in bottom_5:
        i += 1
        image = Image.open(url2filepath(data['inputs']['image']['url']))
        draw = ImageDraw.Draw(image)
        for bbox in data['inputs']['bboxes']:
            left, top = bbox['left'], bbox['top']
            right, bottom = left + bbox['width'], top + bbox['height']
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
        plt.imshow(image)
        plt.axis('off')
        prediction_zero_shot = prediction_data_zero_shot[data['instance_id']]
        score_zero_shot = scores_data_zero_shot[data['instance_id']]
        prediction = prediction_data[data['instance_id']]
        score = scores_data[data['instance_id']]
        prediction_bbox = prediction_data_nobbox[data['instance_id']]
        score_bbox = scores_data_nobbox[data['instance_id']]
        name = str(score) + '-' + prediction[0]
        plt.savefig(f'{name}.png')
        print(prediction, score)
        print(prediction_bbox, score_bbox)
        print(prediction_zero_shot, score_zero_shot)
        print(data['inputs']['clue'])
        
        