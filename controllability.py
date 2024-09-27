import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import bitsandbytes as bnb
import pandas as pd
import random
import transformers
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from huggingface_hub import login

token = os.environ['huggingface_token'] # Load huggingface token from .env to access Llama-2 
login(token=token) # Log into huggingface

model_path = "meta-llama/Llama-2-70b-hf"   # Specify the path to the model
adapter_path = "bryanchrist/MATHWELL"   # Specify the path to the adapter weights

tokenizer = AutoTokenizer.from_pretrained(adapter_path) # Load tokenizer

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
) # Set up bitsandbytes config to load model in 4 bit

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", load_in_4bit=True,
    torch_dtype=torch.bfloat16, use_auth_token=True) # Load model in 4 bit
model = PeftModel.from_pretrained(model, adapter_path) # Create PEFT model 


df = pd.read_csv('data/evaluation_annotations.csv') # Load dataset with annotations for operation
df = df[df['good']==1] # Subset to questions that meet all criteria 
df['output'] = "Question: " + df['question'] + "\n" + "Solution:\n" + df['solution'] # Format into Q/A pairs

# Create DFs for each operation
addition = df.query("addition==1 and total_ops==1")
subtraction = df.query("subtraction==1 and total_ops==1")
multiplication = df.query("multiplication==1 and total_ops==1")
division = df.query("division==1 and total_ops==1")
fractions = df.query("fractions==1 and total_ops==1")
decimals = df.query("decimals==1 and total_ops==1")
multi_ops = df.query("total_ops>1")

# List of topics used for generation 
topics = ['Superman', "Batman", "Wonder Woman", "Barbie", "Power Rangers", "basketball", "soccer", "football", "volleyball", 'field hockey',\
'Fortnite', 'Spiderman', "Iron Man", "Captain America", "Captain Marvel", "Thor, the God of Thunder", "Ninja Turtles", "Black Panther", "Taylor Swift", "swimming",\
"Pokémon", "Super Mario", "Naruto", "unicorns", "Hello Kitty", "Minecraft", "lacrosse", "cheer leading", "LeBron James", "Steph Curry", "Patrick Mahomes",\
"Serena Williams", "dogs", "cats", "dinosaurs", "Harry Potter", "cars", "planes", "trains", "pizza", "cookies", "ice cream", 'candy']

# Function to generate based on a specific operation
def prompt(df, model, tokenizer, n_qs, operation, operator, topics):
    responses = []
    while len(responses)<n_qs:
        topic = random.choice(topics)
        final_prompt = f"Write a grade school math {operation} word problem about {topic} and Python function with a commented out step-by-step solution to solve the word problem. The question you write should only require {operation} to solve, meaning the solution should rely only on use of the {operator} operator."
        prompt = f"Write a grade school math {operation} word problem and Python function with a commented out step-by-step solution to solve the word problem. The question you write should only require {operation} to solve, meaning the solution should rely only on use of the {operator} operator."
        questions = []
        while len(questions)<8:
            question = df['output'].iloc[random.randint(0,len(df)-1)]
            if question not in questions:
                questions.append(question)
        formatted_prompt = []
        for i in range(0,8):
            formatted_prompt.append((f"Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{prompt}\n\n### Response: {questions[i]}"))
        formatted_prompt.append(f"Below is an instruction that describes a task. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{final_prompt}\n\n### Response: ")
        formatted_prompt = "\n".join(formatted_prompt)
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        attention_mask = torch.ones_like(inputs)
        inputs = inputs.to('cuda')
        output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 200, do_sample = True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Split the generated text by the prompt to extract the newly generated part
        generated_text_parts = generated_text.split(final_prompt)
        newly_generated_text = generated_text_parts[-1].strip()
        if "\nBel" in newly_generated_text:
            newly_generated_text = newly_generated_text.split("\nBel")[0]
        if "Solution:" in newly_generated_text:
            if "return" in newly_generated_text:
                responses.append(newly_generated_text)
    return responses

# Generate questions based on each operation, parse them, and save results to a text file
qs = prompt(addition, model, tokenizer, 100, 'addition', "+", topics=topics)
addition_df = pd.DataFrame(qs)
addition_df['operation'] = 'addition'

q_len = len(qs)
add = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " - " not in solution and " * " not in solution and "/" not in solution and "." not in solution and "+" in solution:
            add+=1
            success.append(1)
        else:
            success.append(0)
    except:
        q_len-=1
        
addition_df['success']= success
        
output_file = "controllability.txt"  
with open(output_file, "a") as f: 
    f.write(f"Percentage with just addition: {add/q_len}" + "\n")  

print(f"Percentage with just addition: {add/q_len}")

q_len = len(qs)
add = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " in solution:
            add+=1
    except:
        q_len-=1
print(f"Percentage with addition overall: {add/q_len}")
output_file = "controllability.txt"  
with open(output_file, "a") as f: 
    f.write(f"Percentage with addition overall: {add/q_len}" + "\n") 
    
qs = prompt(subtraction, model, tokenizer, 100, 'subtraction', "-", topics=topics)
subtraction_df = pd.DataFrame(qs)
subtraction_df['operation'] = 'subtraction'

q_len = len(qs)
sub = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " not in solution and " * " not in solution and "/" not in solution and "." not in solution and " - " in solution:
            sub+=1
            success.append(1)
        else:
            success.append(0)
    except:
        q_len-=1
        
subtraction_df['success']= success     

print(f"Percentage with subtraction overall: {sub/q_len}")
output_file = "controllability.txt"  
with open(output_file, "a") as f:  
    f.write(f"Percentage with subtraction only: {sub/q_len}" + "\n") 
    
q_len = len(qs)
sub = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " - " in solution:
            sub+=1
    except:
        q_len-=1
print(f"Percentage with subtraction overall: {sub/q_len}")
output_file = "controllability.txt"  
with open(output_file, "a") as f:  
    f.write(f"Percentage with subtraction overall: {sub/q_len}" + "\n")
    
    
qs = prompt(multiplication, model, tokenizer, 100, 'multiplication', "*", topics=topics)
multiplication_df = pd.DataFrame(qs)
multiplication_df['operation'] = 'multiplication'

q_len = len(qs)
mult = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " not in solution and " - " not in solution and "/" not in solution and "." not in solution and "*" in solution:
            mult+=1
            success.append(1)
        else: 
            success.append(0)
    except:
        q_len-=1

multiplication_df['success']= success    

print(f"Percentage with multiplication only: {mult/q_len}")
output_file = "controllability.txt"  
with open(output_file, "a") as f: 
    f.write(f"Percentage with multiplication only: {mult/q_len}" + "\n")

q_len = len(qs)
mult = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if "*" in solution:
            mult+=1
    except:
        q_len-=1
print(f"Percentage with multiplication overall: {mult/q_len}")
output_file = "controllability.txt"  
with open(output_file, "a") as f:  
    f.write(f"Percentage with multiplication overall: {mult/q_len}" + "\n")
    
qs = prompt(division, model, tokenizer, 100, 'division', "/", topics=topics)
division_df = pd.DataFrame(qs)
division_df['operation'] = 'division'

q_len = len(qs)
div = 0
success = []
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if " + " not in solution and " - " not in solution and " * " not in solution and "." not in solution and "/" in solution:
            div+=1
            success.append(1)
        else: 
            success.append(0)
    except:
        q_len-=1
        
division_df['success']= success 

print(f"Percentage with division only: {div/q_len}")
output_file = "controllability.txt"  
with open(output_file, "a") as f:  
    f.write(f"Percentage with division only: {div/q_len}" + "\n")

q_len = len(qs)
div = 0
for i in range(0, len(qs)):
    try:
        solution = qs[i].split("Solution:")[1]
        if "/" in solution:
            div+=1
    except:
        q_len-=1
print(f"Percentage with division overall: {div/q_len}")
output_file = "controllability.txt" 
with open(output_file, "a") as f: 
    f.write(f"Percentage with division overall: {div/q_len}" + "\n")

# Save all samples to a csv file
all_ops = pd.concat([addition_df, subtraction_df, division_df, multiplication_df])
all_ops.to_csv('controllability_samples.csv')