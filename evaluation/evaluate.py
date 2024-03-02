import json
from pprint import pprint
from bert_score import score as bert_score
from file_utils import VisualGenome_VAL_FILE_PATH, load_json
import sacrebleu
from rouge_score import rouge_scorer
from bart_score import BARTScorer
from data_utils import get_inference, subset_test



def evaluate_inferences(gold_inferences, generated_inferences, device='cuda:0'):
    # Initialize BART scorer
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    
    # Calculate BERTScore
    P, R, F1 = bert_score(generated_inferences, gold_inferences, lang="en", model_type="bert-base-uncased")
    bertscore_results = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}
    
    # Calculate BLEUScore
    bleu_score = sacrebleu.corpus_bleu(generated_inferences, [gold_inferences]).score
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for gold, generated in zip(gold_inferences, generated_inferences):
        scores = scorer.score(gold, generated)
        for key in rouge_scores.keys():
            rouge_scores[key].append(scores[key].fmeasure)
    rouge_scores = {key: sum(values) / len(values) for key, values in rouge_scores.items()}
    
    # Calculate BARTScore
    bart_scores = bart_scorer.score(generated_inferences, gold_inferences, batch_size=4)
    
    return {
        "bertscore": bertscore_results,
        "bleuscore": bleu_score,
        "rougescore": rouge_scores,
        "bartscore": bart_scores,
    }

# # Example usage
# inference_pairs = [
#     ("The cat sits on the mat.", "A cat is on the mat."),
#     ("They are eating a meal.", "A meal is being eaten by them."),
# ]

# results = evaluate_inferences(inference_pairs)
# print(results)
def post_process_results(text):
    return text.strip().split("\n")[0]

if __name__ == "__main__":
    base_results_path = "/data/tir/projects/tir4/users/svarna/Sherlock/data/results/unimodal_baselines/text/inference"
    paths = {
        "T5": f"{base_results_path}/t5_base_inference_with_instruction_10_tokens.json",
        "FLAN_T5": f"{base_results_path}/flant5_base_inference_with_instruction_10_tokens.json",
        "LLAMA_7B": f"{base_results_path}/llama_7b_inference_with_instruction_10_tokens.json",
        "LLAMA_70B": f"{base_results_path}/llama_70b_inference_with_instruction_10_tokens.json"
    }

    gold_val_data = json.load(open(subset_test))
    gold_inferences = [get_inference(x) for x in gold_val_data]

    
    models_results = {}
    for model, path in paths.items():
        data = load_json(path)
        data = [post_process_results(d) for d in data]
        print(model, len(gold_inferences), len(data))
        models_results[model] = evaluate_inferences(gold_inferences, data)

        pprint(models_results)

    flattened_results = []

    for model_name, results in models_results.items():
        flattened = {
            'model': model_name,
            'bertscore_precision': results['bertscore']['precision'],
            'bertscore_recall': results['bertscore']['recall'],
            'bertscore_f1': results['bertscore']['f1'],
            'bleuscore': results['bleuscore'],
            'rougescore_rouge1': results['rougescore']['rouge1'],
            'rougescore_rouge2': results['rougescore']['rouge2'],
            'rougescore_rougeL': results['rougescore']['rougeL'],
            'bartscore_mean': sum(results['bartscore']) / len(results['bartscore']),
        }
        flattened_results.append(flattened)

    import pandas as pd

    df = pd.DataFrame(flattened_results)
    df.to_csv('model_evaluation_results.csv', index=False)




