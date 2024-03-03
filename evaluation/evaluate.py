import json
from pprint import pprint
from statistics import mean
from bert_score import score as bert_score
from file_utils import VisualGenome_VAL_FILE_PATH, load_json, subset_test
import sacrebleu
from rouge_score import rouge_scorer
from bart_score import BARTScorer
from data_utils import get_inference



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
        "bartscore": mean(bart_scores)
    }

# # Example usage
# inference_pairs = [
#     ("The cat sits on the mat.", "A cat is on the mat."),
#     ("They are eating a meal.", "A meal is being eaten by them."),
# ]

# results = evaluate_inferences(inference_pairs)
# print(results)
def post_process_results(text):
    text = "".join(text.split("Abductive Inference:")[2:3])
    return text.strip().split("\n")[0]

if __name__ == "__main__":

    # inference_pairs = [
    # ("The cat sits on the mat.", "A cat is on the mat."),
    # ("They are eating a meal.", "A meal is being eaten by them."),
    # ]

    # gt = ["The cat sits on the mat.", "They are eating a meal."]
    # original = ["A cat is on the mat.", "A meal is being eaten by them."]

    # results = evaluate_inferences(gt, original)
    # print(results)


    base_results_path = "/data/tir/projects/tir4/users/svarna/Sherlock/data/results/unimodal_baselines/text/inference"
    base_results_path2 = "/data/tir/projects/tir4/users/svarna/Sherlock/data/results/unimodal_baselines/text/few_shot_inference"
    paths = {
        # "T5": f"{base_results_path}/t5_base_subset_val_10_tokens.json",
        # "FLAN_T5": f"{base_results_path}/flanT5_subset_val_10_tokens.json",
        # "LLAMA_7B": f"{base_results_path}/llama_7b_subset_val_10_tokens.json",
        # "LLAMA_70B": f"{base_results_path}/llama_70b_subset_val_10_tokens.json",
        # "few_shot_T5": f"{base_results_path2}/t5_base_subset_val_15_tokens1.json",
        # "few_shot_FLAN_T5": f"{base_results_path2}/flanT5_subset_val_10_tokens.json",
        # "few_shot_LLAMA_7B": f"{base_results_path2}/llama_7b_subset_val_10_tokens.json",
        # "few_shot_LLAMA_70B": f"{base_results_path2}/llama_70b_subset_val_10_tokens.json",
        # "finetuned_flanT5": "/data/tir/projects/tir4/users/svarna/Sherlock/data/results/unimodal_baselines/text/inference/flanT5_finetuned_subset_val_10_tokens1.json",
        "few_shot_gpt2": "/data/tir/projects/tir4/users/svarna/Sherlock/data/results/unimodal_baselines/text/few_shot_inference/gpt2_subset_val_15_tokens.json"

    }

    gold_val_data = json.load(open(subset_test))
    gold_inferences = [get_inference(x) for x in gold_val_data]

    
    models_results = {}
    for model, path in paths.items():
        data = load_json(path)
        if isinstance(data[0], list):
            data = sum(data, [])
        data = [post_process_results(d) for d in data]
        if len(gold_inferences) != len(data):
            continue
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
            'bartscore_mean': results['bartscore'],
        }
        flattened_results.append(flattened)

    import pandas as pd

    df = pd.DataFrame(flattened_results)
    df.to_csv('model_evaluation_results.csv', index=False)


    # models_results = {'FLAN_T5': {'bartscore': -4.9054614573939945,
    #             'bertscore': {'f1': 0.48516178131103516,
    #                         'precision': 0.4808981418609619,
    #                         'recall': 0.4922444522380829},
    #             'bleuscore': 1.3694743906382425,
    #             'rougescore': {'rouge1': 0.21691404329174313,
    #                             'rouge2': 0.04129334953695967,
    #                             'rougeL': 0.19333509581573866}},
    # 'LLAMA_70B': {'bartscore': -4.391004876575937,
    #             'bertscore': {'f1': 0.5420212149620056,
    #                             'precision': 0.5501251816749573,
    #                             'recall': 0.5377285480499268},
    #             'bleuscore': 5.712077313473758,
    #             'rougescore': {'rouge1': 0.2944154058726904,
    #                             'rouge2': 0.11233668309467504,
    #                             'rougeL': 0.2746135261235311}},
    # 'LLAMA_7B': {'bartscore': -4.608444012158638,
    #             'bertscore': {'f1': 0.4958929419517517,
    #                             'precision': 0.4929350018501282,
    #                             'recall': 0.5021072626113892},
    #             'bleuscore': 2.8481101762322756,
    #             'rougescore': {'rouge1': 0.23534264139231792,
    #                             'rouge2': 0.060703685643557374,
    #                             'rougeL': 0.2145483307705079}},
    # 'T5': {'bartscore': -4.97773146659044,
    #         'bertscore': {'f1': 0.40758055448532104,
    #                     'precision': 0.400209903717041,
    #                     'recall': 0.41738879680633545},
    #         'bleuscore': 0.6970733664084848,
    #         'rougescore': {'rouge1': 0.13122105269916381,
    #                     'rouge2': 0.014541929135321452,
    #                     'rougeL': 0.1193841709853379}},
    # 'few_shot_FLAN_T5': {'bartscore': -4.918994580907672,
    #                     'bertscore': {'f1': 0.4876180589199066,
    #                                     'precision': 0.4851573407649994,
    #                                     'recall': 0.49290430545806885},
    #                     'bleuscore': 1.450042713342972,
    #                     'rougescore': {'rouge1': 0.22135294405656392,
    #                                     'rouge2': 0.0415880362851632,
    #                                     'rougeL': 0.19768742852735854}},
    # 'few_shot_LLAMA_70B': {'bartscore': -4.3574607897942546,
    #                         'bertscore': {'f1': 0.5377777814865112,
    #                                     'precision': 0.5512829422950745,
    #                                     'recall': 0.5282019972801208},
    #                         'bleuscore': 3.490174869255437,
    #                         'rougescore': {'rouge1': 0.23156770456752523,
    #                                     'rouge2': 0.06489735102572794,
    #                                     'rougeL': 0.2189653872928533}},
    # 'few_shot_LLAMA_7B': {'bartscore': -4.519109925043546,
    #                     'bertscore': {'f1': 0.5020552277565002,
    #                                     'precision': 0.5098420977592468,
    #                                     'recall': 0.4974380433559418},
    #                     'bleuscore': 2.063256516135322,
    #                     'rougescore': {'rouge1': 0.18971255231217243,
    #                                     'rouge2': 0.043878493322123124,
    #                                     'rougeL': 0.17773234084247855}},          
    #     'few_shot_T5': {'bartscore': -5.107169607206098,
    #                 'bertscore': {'f1': 0.37563928961753845,
    #                             'precision': 0.3521600663661957,
    #                             'recall': 0.40457820892333984},
    #                 'bleuscore': 0.13884371725231465,
    #                 'rougescore': {'rouge1': 0.09076782129404426,
    #                                 'rouge2': 0.0022530201419105993,
    #                                 'rougeL': 0.08728187343774525}}}

    # flattened_results = []

    # for model_name, results in models_results.items():
    #     flattened = {
    #         'model': model_name,
    #         'bertscore_precision': round(results['bertscore']['precision'], 2),
    #         'bertscore_recall': round(results['bertscore']['recall'], 2),
    #         'bertscore_f1': round(results['bertscore']['f1']*100, 2),
    #         'bleuscore': round(results['bleuscore'], 2),
    #         'rougescore_rouge1': round(results['rougescore']['rouge1'], 2),
    #         'rougescore_rouge2': round(results['rougescore']['rouge2'], 2),
    #         'rougescore_rougeL': round(results['rougescore']['rougeL']*100, 2),
    #         'bartscore_mean': round(results['bartscore'], 2),
    #     }
    #     flattened_results.append(flattened)

    # import pandas as pd

    # df = pd.DataFrame(flattened_results)
    # df.to_csv('model_evaluation_results.csv', index=False)


