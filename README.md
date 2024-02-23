# MATHWELL
MATHWELL: Generating Educational Math Word Problems at Scale

[Paper]()| [Model Weights](https://huggingface.co/bryanchrist/MATHWELL) | [SGSM Dataset](https://huggingface.co/datasets/bryanchrist/SGSM)| [Human Annotation Dataset](https://huggingface.co/datasets/bryanchrist/annotations)

# Overview 
Math word problems are critical K-8 educational tools, but writing them is time-consuming and requires domain expertise. We suggest that language models can support K-8 math education by automatically generating problems at scale. To be educational, generated problems must be 1) solvable, 2) accurate, and 3) appropriate. Existing datasets are unlabeled for these criteria, making them ill-suited for training problem generators. We introduce MATHWELL, a Llama-2 (70B) model iteratively finetuned to generate K-8 math word problems using data from expert annotation. Using MATHWELL, we generate the largest English word problem dataset to date, containing 20,490 problems. 3,484 are scored by domain experts who find MATHWELL has a 40% higher share of problems that have executable solutions and meet all criteria than alternatives, with 74% of its problems with executable solutions being solvable, accurate, and appropriate.

# License and Intended Use
Our datasets, model and code are released under the MIT license. MATHWELL falls under the Llama-2 license and require access to the 70B model weights through a HuggingFace access token. 

# Getting Started
After installing PyTorch (follow instructions [here](https://pytorch.org/get-started/locally/)), to install the dependencies for this codebase, you can run: 
```bash
pip install -U -r requirements.txt
```
# Interacting with MATHWELL
generate.py is an example script for loading and interacting with the MATHWELL model. In order to run the script, you'll need a .env file with a Llama-2 70B Hugging Face access token called huggingface_token. You can edit generate.py based on what precision you want to load the model in, what topics you want MATHWELL to generate questions about, and how many questions you want to generate. The default is to load the model in 4 bits and generate 10 questions about topics randomly selected from the list of topics we used to generate SGSM. The script will automatically parse the model outputs for executable code solutions and continue generation until the model has output the desired number of questions. The script will conclude by outputing a csv file titled "questions.csv" that contains the questions, solutions, answers, and associated topics. 

