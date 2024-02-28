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

df = pd.read_csv('data/sgsm.csv') # Load SGSM dataset for few-shot prompting
df = df[df['subset']=="sgsm_train"] # Subset SGSM to verified training subset
df['output'] = "Question: " + df['question'] + "\n" + "Solution:\n" + df['solution'] # Create a column in df that contains the question and solution together for few-shot prompting below

# List of topics used to generate SGSM - You can swap this out with any desired topic
topics = ['Superman', "Batman", "Wonder Woman", "Barbie", "Power Rangers", "basketball", "soccer", "football", "volleyball", 'field hockey',\
'Fortnite', 'Spiderman', "Iron Man", "Captain America", "Captain Marvel", "Thor, the God of Thunder", "Black Panther", "Taylor Swift", "swimming",\
"Pokémon", "Super Mario", "Naruto", "unicorns", "Hello Kitty", "Minecraft", "lacrosse", "cheer leading", "LeBron James", "Steph Curry", "Patrick Mahomes",\
"Serena Williams", "dogs", "cats", "dinosaurs", "Harry Potter", "cars", "planes", "trains", "pizza", "cookies", "ice cream", 'candy',\
    "Frozen (Elsa and Anna)",
    "Minecraft",
    "Star Wars",
    "Paw Patrol",
    "My Little Pony",
    "Minions",
    "Jurassic Park",
    "SpongeBob SquarePants",
    "Disney Princesses",
    "Toy Story",
    "The Incredibles",
    "Scooby-Doo",
    "Peppa Pig",
    "Dora the Explorer",
    "Pikachu",
    "Thomas the Tank Engine",
    "Sonic the Hedgehog",
    "Transformers",
    "Minions",
    "Cinderella",
    "Moana",
    "Shrek",
    "Winnie the Pooh",
    "Tom and Jerry",
    "Sesame Street",
    "The Lion King",
    "Alice in Wonderland",
    "The Little Mermaid",
    "Peter Pan",
    "Aladdin",
    "The Jungle Book",
    "Pocahontas",
    "Beauty and the Beast",
    "Frozen",
    "Ratatouille",
    "Finding Nemo",
    "Cars",
    "Up",
    "The Simpsons",
    "Looney Tunes",
    "Teenage Mutant Ninja Turtles",
    "Scooby-Doo",
    "Mythical Creatures (dragons, unicorns)",
    "Dinosaurs",
    "Space and Astronauts",
    "Robots",
    "Aliens",
    "Exploring the Ocean",
    "Underwater Creatures",
    "Pirates",
    "Fairies",
    "Wizards",
    "Magic Tricks",
    "Time Travel",
    "Detectives and Mystery",
    "Inventions",
    "The Avengers",
    "The Justice League",
    "Dance and Ballet",
    "Music Instruments",
    "Art and Drawing",
    "Science Experiments",
    "Cooking and Baking",
    "DIY Crafts",
    "Board Games",
    "Puzzles",
    "Riddles",
    "Pets (cats, dogs, hamsters)",
    "Farm Animals",
    "Zoo Animals",
    "Wildlife Conservation",
    "Plants and Gardening",
    "Hiking and Nature",
    "Weather and Meteorology",
    "The Solar System",
    "Camping",
    "National Parks",
    "Trains and Railroads",
    "Planes and Aviation",
    "Cars and Racing",
    "Construction Vehicles",
    "Firefighters",
    "Police Officers",
    "Doctors and Nurses",
    "Astronauts and Space Exploration",
    "Animals and Wildlife",
    "Space and Astronomy",
    "Robots and Technology",
    "Underwater Life",
    "Fairy Tales and Folklore",
    "Science Experiments",
    "Outer Space",
    "Weather and Meteorology",
    "Art and Drawing",
    "Music and Instruments",
    "Cooking and Baking",
    "Insects and Bugs",
    "Historical Figures",
    "Countries and Cultures",
    "Mythical Creatures",
    "Magic and Wizards",
    "Friendship and Relationships",
    "Ocean Life",
    "Cars and Vehicles",
    "Famous Inventors",
    "Famous Artists",
    "Ancient Civilizations",
    "Space Exploration",
    "DIY Crafts",
    "Gardening",
    "Environmental Conservation",
    "Time Travel",
    "Pirates and Treasure",
    "Famous Scientists",
    "Computer Programming",
    "Unexplained Mysteries",
    "Planets and the Solar System",
    "Cartoons and Animated Shows",
    "Photography",
    "National Parks",
    "Dance and Ballet",
    "Board Games",
    "Books and Reading",
    "Volcanoes",
    "Mythology",
    "Ancient Egypt",
    "Reptiles and Amphibians",
    "Recycling",
    "Fairy Gardens",
    "Indoor Games",
    "Marine Biology",
    "Virtual Reality",
    "Natural Disasters",
    "Construction and Building",
    "Inventions",
    "the Circus and Performing Arts",
    "Science Fiction",
    "Pottery and Ceramics",
    "Famous Explorers",
    "Birds and Bird Watching",
    "Famous Landmarks",
    "Health and Nutrition",
    "Myths and Legends",
    "Fashion and Clothing",
    "DIY Science Projects",
    "Cultural Festivals",
    "Construction Vehicles",
    "Forests and Trees",
    "Mummies",
    "Famous Composers",
    "Circus Animals",
    "Geology",
    "Farm Life",
    "Travel and Adventure",
    "Ballet and Dance",
    "Native American Culture",
    "Whales and Dolphins",
    "Mystery Stories",
    "Hiking and Camping",
    "Games and Puzzles",
    "Space Aliens and UFOs"
]

n_questions = 10 # Specify the number of questions you want to generate
generated_questions = 0 # Start counter for the number of generated questions with executable code
data = [] # Blank list to store questions with executable code
while generated_questions < n_questions:
    topic = random.choice(topics) # Select a random topic

    # Format the prompt
    final_prompt = f"Write a grade school math word problem about {topic} and Python function with a commented out step-by-step solution to solve the word problem."
    prompt = "Write a grade school math word problem and Python function with a commented out step-by-step solution to solve the word problem."
    questions = []
    for i in range(0, 8):
        question = df['output'].iloc[random.randint(0,len(df)-1)]
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

    #Query the model 
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 250, do_sample = True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the generated text by the prompt to extract the newly generated part
    generated_text_parts = generated_text.split(final_prompt)
    newly_generated_text = generated_text_parts[-1].strip()
    if "\nBel" in newly_generated_text:
        newly_generated_text = newly_generated_text.split("\nBel")[0] # Split up a generation that contains more than one question
    
    # Parse the question for executable code
    try:
        section = newly_generated_text.split("### Response:")[1].strip()
        question = section.split("Question:")[1].strip()
        question = question.split("Solution:")[0].strip()
        solution = section.split("Solution:")[1].strip()
        if "##" in question:
            question = question.split("##")[0]
        if "\nBel" in solution:
            solution = solution.split("\nBel")[0]
        if "##" in solution:
            solution = solution.split("##")[0]
        if "return" not in solution: # Try to pick up generations that calculated a result but did not include a return statement
            try: 
                solution = solution + '\n' + '    return result'
                solution_text = solution
                try:
                    # Execute the Python code
                    exec(solution)
                    answer = solution()
                    if answer==None:
                        continue
                    generated_questions+=1 # Update the counter
                    data.append({"question": question, "solution": solution_text, 'answer': answer, 'topic': topic}) # Append the question, solution, answer, and topic to the list
                except: 
                    try: # Pick up generations that did not indent the code block
                        solution_new = ""
                        lines = solution.split('\n')
                        indent = False
                        for line in lines:
                            if line.startswith('def solution():'):
                                solution_new += line + '\n'

                            if not line.startswith('def solution():') and not line.startswith('    return result'):
                                solution_new += '    ' + line + '\n'

                            if line.startswith('    return result'):
                                solution_new += line
                        solution_text = solution_new
                        # Execute the Python code
                        exec(solution_new)
                        answer = solution()
                        if answer==None:
                            continue
                        generated_questions+=1 # Update the counter
                        data.append({"question": question, "solution": solution_text, 'answer': answer, "topic": topic}) # Append the question, solution, answer, and topic to the list
                    except:
                            pass # New generation if codeblock fails
            except:
                pass # New generation if codeblock fails
        if "return" in solution: # Pick up generations with proper formatting
            try: 
                solution_text = solution
                try:
                    # Execute the Python code
                    exec(solution)
                    answer = solution()
                    if answer==None:
                        continue
                    generated_questions+=1 # Update the counter
                    data.append({"question": question, "solution": solution_text, 'answer': answer, "topic": topic}) # Append the question, solution, answer, and topic to the list
                except: 
                    try: # Pick up generations that did not indent the code block
                        solution_new = ""
                        lines = solution.split('\n')
                        indent = False
                        for line in lines:
                            if line.startswith('def solution():'):
                                solution_new += line + '\n'

                            if not line.startswith('def solution():') and not line.startswith('    return result'):
                                solution_new += '    ' + line + '\n'

                            if line.startswith('    return result'):
                                solution_new += line
                        solution_text = solution_new
                        # Execute the Python code
                        exec(solution_new)
                        answer = solution()
                        if answer==None:
                            continue
                        generated_questions+=1 # Update the counter
                        data.append({"question": question, "solution": solution_text, 'answer': answer, "topic": topic}) # Append the question, solution, answer, and topic to the list
                    except:
                        pass # New generation if codeblock fails
            except:
                pass # New generation if codeblock fails
    except:
        pass # New generation if codeblock fails
        
df1 = pd.DataFrame.from_dict(data)
df1.to_csv('questions.csv', index=False)
        