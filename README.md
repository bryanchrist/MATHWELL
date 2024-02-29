# MATHWELL

[Paper](https://arxiv.org/abs/2402.15861) | [Model Weights](https://huggingface.co/bryanchrist/MATHWELL) | [SGSM Dataset](https://huggingface.co/datasets/bryanchrist/SGSM) | [Human Annotation Dataset](https://huggingface.co/datasets/bryanchrist/annotations)

This codebase is based off the paper MATHWELL: Generating Educational Math Word Problems at Scale, and includes code for interacting with the MATHWELL model and SGSM dataset. 

# Overview 
Math word problems are critical K-8 educational tools, but writing them is time-consuming and requires domain expertise. We suggest that language models can support K-8 math education by automatically generating problems at scale. To be educational, generated problems must be 1) solvable, 2) accurate, and 3) appropriate. Existing datasets are unlabeled for these criteria, making them ill-suited for training problem generators. We introduce MATHWELL, a Llama-2 (70B) model iteratively finetuned to generate K-8 math word problems using data from expert annotation. Using MATHWELL, we generate the largest English word problem dataset with Program of Thought (PoT) rationales to date, containing 20,490 problems. 3,484 are scored by domain experts who find MATHWELL has a 40% higher share of problems that have executable solutions and meet all criteria than alternatives, with 74% of its problems with executable solutions being solvable, accurate, and appropriate. We release our model, data, and annotations.

# License and Intended Use
Our datasets, model and code are released under the GNU GPLv3 license. MATHWELL falls under the Llama-2 license and requires access to the 70B model weights through a HuggingFace access token. 

# Getting Started
After installing PyTorch (follow instructions [here](https://pytorch.org/get-started/locally/)), to install the dependencies for this codebase, you can run: 
```bash
pip install -U -r requirements.txt
```

For issues with loading MATHWELL using bitsandbytes, refer to their [documentation](https://github.com/TimDettmers/bitsandbytes). 

# Loading MATHWELL
You can load the MATHWELL adapter weights for 4-bit Llama-2 (70B) by running the following code (assuming you have a .env file that specifies a HuggingFace token to access Llama-2 defined as huggingface_token): 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
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
```

# Loading SGSM and the Human Annotation Dataset 
You can load our datasets using the following code: 
```python
from datasets import load_dataset
sgsm = load_dataset('bryanchrist/sgsm')
annotations = load_dataset('bryanchrist/annotations')
```
*Note: You will need to create a train/test split if you wish to have a validation sample. 

# Interacting with MATHWELL
generate.py is an example script for loading and interacting with the MATHWELL model, including the suggested few-shot prompting strategy using data from SGSM Train in an Alpaca-style prompt. In order to run the script, you'll need a .env file with a Llama-2 70B HuggingFace access token called huggingface_token. You can edit generate.py based on what precision you want to load the model in, what topics you want MATHWELL to generate questions about, and how many questions you want to generate. The default is to load the model in 4 bits and generate 10 questions about topics randomly selected from the list of topics we used to generate SGSM. The script will automatically parse the model outputs for executable code solutions and continue generation until the model has output the desired number of questions. The script will conclude by outputing a csv file titled "questions.csv" that contains the questions, solutions, answers, and associated topics. 

# Training MATHWELL
qlora.py contains the code to train the MATHWELL model and is based off the [QLoRA repository](https://github.com/artidoro/qlora) with minor changes to support loading a HuggingFace access token and fixing an embedding dimension issue when using Llama-2 by padding to a multiple of 64. It supports all the functionality listed in their repository. In order to run the script, you'll need a .env file with a Llama-2 70B HuggingFace access token called huggingface_token. To finetune the model the same way we did in the paper, you'll need to do two rounds of finetuning. 

First, you will finetune using existing Math QA data (MathInstruct GSM8K, MATH, and Theorem QA PoT datasets) for 4,250 steps. This can be accomplished by running: 

```bash
python qlora.py \
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --output_dir mathwell \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 250 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1000 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 400 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset="data/initial_train.json" \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 4250 \
    --eval_steps 187 \
    --learning_rate 0.00001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0001 \
    --seed 0
```

Next, you will conduct a second round of finetuning using SGSM Train for 1,250 steps. This can be accomplished by running: 

```bash
python qlora.py \
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --output_dir mathwell \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 250 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 285 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 400 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset="data/sgsm_train.json" \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 1250 \
    --eval_steps 187 \
    --learning_rate 0.00001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0001 \
    --seed 0
```

# Automatically Evaluating Outputs
You can find sample scripts for automatically evaluating model outputs and training text classifiers in the main directory (text_classifier.py, question_length.py, ppl.py, reading_level.py, bertscore.py). These files contain the same code we used to conduct experiments in the paper. They each contain functions to conduct the experiment and an example implementation. You can modify them to run any of the experiments reported in our paper. 

# Citation
```bash
@misc{christ2024mathwell,
      title={MATHWELL: Generating Educational Math Word Problems at Scale}, 
      author={Bryan R Christ and Jonathan Kropko and Thomas Hartvigsen},
      year={2024},
      eprint={2402.15861},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
