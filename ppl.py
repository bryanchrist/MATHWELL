import torch
import os
import pandas as pd
import numpy as np 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from huggingface_hub import login

token = os.environ['huggingface_token'] # Load huggingface token from .env to access Llama-2 
login(token=token) # Log into huggingface

# Load Llama-2 70B in 4 bit
model_path = "meta-llama/Llama-2-70b-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    load_in_4bit=True,   
    trust_remote_code=True,
    token=token,
    device_map="auto",
)

# Load data
egsm = pd.read_csv('data/egsm.csv')

# This function calculates the perplexity for question/answer pairs in a pandas dataframe in the format outputted by MATHWELL. Note that you will need a DF with a separate question
# and solution column for this code to run. 
def perplexity(df):
    # Create list to store perplexities
    ppls = []

    # Loop over all question/answer pairs in df, calculating ppl for each
    for i in range(0, len(df)):
        text = "Question: " + str(df.iloc[i]['question']) + "\n" + "Solution:\n" + str(df.iloc[i]['solution'])
        inputs = tokenizer(text, return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss)
        ppl = ppl.cpu().detach().numpy()
        ppls.append(ppl)
    return ppls

# Example usage
ppl = perplexity(egsm)
print(f'Average EGSM overall perplexity: {np.mean(ppl)} Standard Deviation: {np.std(ppl)}')
