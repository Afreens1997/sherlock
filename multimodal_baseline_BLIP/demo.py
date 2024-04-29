from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoProcessor, Blip2ForConditionalGeneration
import torch


processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

img_path = 'sample1.png'

image = Image.open(img_path)
prompt = "Question: Describe the figure in detail. Answer:"
inputs = processor(images=image, return_tensors="pt").to(device, dtype=torch.float16)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(generated_text)