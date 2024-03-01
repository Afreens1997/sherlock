import evaluate
import json
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer


# BERTScore calculation

DATA_PATH = "../../../../data/tir/projects/tir4/users/svarna/Sherlock/data"

with open(f"{DATA_PATH}/multimodal_baselines/inferences/BLIP_zero_shot.json", "r") as file:
    inferences = json.load(file)

predictions= []
references = []
for data in inferences:
    predictions.append(inferences[data][0])
    references.append(inferences[data][1])

# print(len(predictions), len(references))

bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print(results)

rouge = evaluate.load('rouge')
results = rouge.compute(predictions=predictions, references=references)
print(results)

# bert = evaluate.load('bertscore')
# results = bert.compute(predictions=predictions, references=references, lang='en')
# print(len(results['precision']))

scorer = BERTScorer(model_type='bert-base-uncased')
P, R, F1 = scorer.score(predictions, references)
print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")