import matplotlib.pyplot as plt

color_map = {
    "Zero Shot": "skyblue",
    "Fine Tuned - only Image": "lightgreen",
    "Fine Tuned - Image + BBox": "lightgreen",
    "Fine Tuned - Image + BBox + Clue": "salmon"
}

def apply_colors(scores):
    return [color_map[model] for model in scores.keys()]

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(y[i]/2, i, y[i], ha = 'center')

def save_plots_5x5(scores, metric_name, color_map, file_name):
    colors = apply_colors(scores)
    plt.figure(figsize=(5, 3))  # Set figure size to 5x5
    plt.barh(list(scores.keys()), list(scores.values()), color=colors, height=0.5)  # Horizontal bar chart with reduced bar width
    plt.xlabel('Score')
    plt.tight_layout()
    addlabels(list(scores.keys()), list(scores.values()))
    plt.savefig(f'{file_name}_5x5.png')
    plt.close()

model_scores = {
    "Zero Shot": {"BLEU-4": 0.42, "ROUGE-L": 10.73, "BERT-F1": 40.88},
    "Fine Tuned - only Image": {"BLEU-4": 2.37, "ROUGE-L": 22.24, "BERT-F1": 49.03},
    "Fine Tuned - Image + BBox": {"BLEU-4": 2.56, "ROUGE-L": 22.41, "BERT-F1": 49.40},
    "Fine Tuned - Image + BBox + Clue": {"BLEU-4": 4.08, "ROUGE-L": 25.70, "BERT-F1": 60.20},
}

sorted_model_scores = {
    "Zero Shot": model_scores["Zero Shot"],
    "Fine Tuned - Image + BBox + Clue": model_scores["Fine Tuned - Image + BBox + Clue"],
    "Fine Tuned - only Image": model_scores["Fine Tuned - only Image"],
    "Fine Tuned - Image + BBox": model_scores["Fine Tuned - Image + BBox"],
}

sorted_bleu_scores = {model: scores["BLEU-4"] for model, scores in sorted_model_scores.items()}
sorted_rouge_scores = {model: scores["ROUGE-L"] for model, scores in sorted_model_scores.items()}
sorted_bert_scores = {model: scores["BERT-F1"] for model, scores in sorted_model_scores.items()}

# Generate and save plots with 5x5 size without captions
save_plots_5x5(sorted_bleu_scores, "BLEU-4", color_map, "BLEU-4_ScoresBLIP")
save_plots_5x5(sorted_rouge_scores, "ROUGE-L", color_map, "ROUGE-L_ScoresBLIP")
save_plots_5x5(sorted_bert_scores, "BERT-F1", color_map, "BERT-F1_ScoresBLIP")