import asyncio
import nest_asyncio
# import text_generation as tg
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast

import json
from file_utils import RESULTS_FOLDER, subset_test, save_json
from data_utils import get_clue, get_inference, get_instance_id
from transformers import GPT2Tokenizer, TFGPT2Model
from transformers import pipeline, set_seed



time_map = {} 
class Reader:
    def __init__(self, hosted_api_endpoint=None, tokenizer=None, model=None):

        self.hosted_api_endpoint = hosted_api_endpoint
        if self.hosted_api_endpoint:
            # initialize async text generation inference client
            nest_asyncio.apply()
            self.async_client = tg.AsyncClient(self.hosted_api_endpoint)
        else:
            self.tokenizer = tokenizer
            self.model = model
        

    async def batch_generate(self, texts, max_new_tokens=20, truncate=2000):
        return await asyncio.gather(*[self.async_client.generate(text, max_new_tokens=max_new_tokens, truncate=truncate) for text in texts])

    def generate(self, prompts, max_new_tokens=10, truncate=2000):
        if self.hosted_api_endpoint:
            responses = asyncio.run(self.batch_generate(prompts, max_new_tokens, truncate))
            return [r.generated_text for r in responses]
        else:
            print(len(prompts))
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
            outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
            return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    

def get_baseline_inference_from_clues(model_name, model=None, tokenizer=None, model_endpoint=None, batch_size=10, max_new_tokens=10):
    model = Reader(hosted_api_endpoint=model_endpoint, model=model, tokenizer=tokenizer )
    example_clue = "a hot dog in his hand"
    example_inference = "he's about to eat the hot dog"
    val_data = json.load(open(subset_test))
    print(len(val_data))
    results = []
    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i+batch_size]
        clues_batch = [f"Generate abductive inference based on the clue. \nClue: {example_clue} Abductive Inference: {example_inference}\nClue: {get_clue(x)} Abductive Inference: " for x in batch]
        model_inferences = model.generate(clues_batch, max_new_tokens=max_new_tokens)
        results.extend(model_inferences)
        save_json(results, f"{RESULTS_FOLDER}/unimodal_baselines/text/few_shot_inference/{model_name}_subset_val_{max_new_tokens}_tokens1.json")

def get_baseline_inference_from_clues_gpt2(model_name, model=None, tokenizer=None, model_endpoint=None, batch_size=10, max_new_tokens=10):
    model = Reader(hosted_api_endpoint=model_endpoint, model=model, tokenizer=tokenizer )
    example_clue = "a hot dog in his hand"
    example_inference = "he's about to eat the hot dog"
    val_data = json.load(open(subset_test))
    set_seed(42)
    print(len(val_data))
    results = []
    generator = pipeline('text-generation', model='gpt2')
    for i in range(0, len(val_data),batch_size):
        batch = val_data[i:i+batch_size]
        clues_batch = [f"Generate abductive inference based on the clue. \nClue: {example_clue} Abductive Inference: {example_inference}\nClue: {get_clue(x)} Abductive Inference: " for x in batch]
        model_inferences = model.generate(clues_batch, max_new_tokens=max_new_tokens)
        results.extend(model_inferences)
        save_json(results, f"{RESULTS_FOLDER}/unimodal_baselines/text/few_shot_inference/{model_name}_subset_val_{max_new_tokens}_tokens.json")

        # dp = val_data[i]
        # clues_batch = f"Generate abductive inference based on the clue. \nClue: {example_clue} Abductive Inference: {example_inference}\nClue: {get_clue(dp)} Abductive Inference: "
        # model_inferences = generator(clues_batch, max_length=len(clues_batch) + 15, num_return_sequences=1)
        # print(i)
        # results.extend([x["generated_text"][len(clues_batch):] for x in model_inferences])
        # save_json(results, f"{RESULTS_FOLDER}/unimodal_baselines/text/few_shot_inference/{model_name}_subset_val_{max_new_tokens}_tokens1.json")


# get_baseline_inference_from_clues("llama_7b", model_endpoint=f"http://inst-0-35:8000/", max_new_tokens=10)
# get_baseline_inference_from_clues("llama_7b", model_endpoint=f"http://inst-0-35:8000/", max_new_tokens=20)
# get_baseline_inference_from_clues("llama_70b", model_endpoint=f"http://babel-1-27:7103/", max_new_tokens=10)
# get_baseline_inference_from_clues("llama_70b", model_endpoint=f"http://babel-1-27:7103/", max_new_tokens=20)




if __name__ == "__main__":

    # get_baseline_inference_from_clues("llama_7b", model_endpoint=f"http://babel-8-11:8000/", max_new_tokens=10)
    # get_baseline_inference_from_clues("llama_70b", model_endpoint=f"http://babel-4-36:7103/", max_new_tokens=10)
    # get_baseline_inference_from_clues("flanT5", model_endpoint=f"http://babel-5-19:9428/", max_new_tokens=10)
      
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = TFGPT2Model.from_pretrained('gpt2')

    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = "[PAD]"
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    get_baseline_inference_from_clues_gpt2("gpt2", model=model, tokenizer=tokenizer, max_new_tokens=15)


    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to("cuda")
    # get_baseline_inference_from_clues_gpt2("gpt2", model=model, tokenizer=tokenizer, max_new_tokens=15)
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    # get_baseline_inference_from_clues("flant5_base", model=model, tokenizer=tokenizer, max_new_tokens=10)

