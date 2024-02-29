from transformers import AutoTokenizer 
import pandas as pd
import torch
import numpy as np
import os
from huggingface_hub import login

token = os.environ['huggingface_token'] #Load huggingface token from .env to access Llama-2 
login(token=token) #Log into huggingface

sgsm = pd.read_csv('data/sgsm.csv')
model_path = "meta-llama/Llama-2-70b-hf"   # Specify the path to the model
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)

# This function calculates the length for questions in a pandas dataframe based on the Llama-2 tokenizer. Note you need to specify the column name for the question text.
def check_length(df, varname):
    # Create list to store lengths
    question_lengths = []

    # Loop over all questions, calculating the length for each
    for i in range(0, len(df)):
        output = df.iloc[i][varname]
        try: 
            inputs = tokenizer.encode(output, return_tensors="pt")
        except:
            pass
        length = inputs.shape[1]
        question_lengths.append(length)
    return question_lengths

# Example usage 
sgsm_len = check_length(sgsm, 'question')
print(f'Average overall question length in tokens: {np.mean(sgsm_len)} Standard Deviation: {np.std(sgsm_len)}')