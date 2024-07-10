import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# GROQ AI Import
from groq import Groq
from authtoken import groq_token, folder_path # authtoken and folder_path are unique to your API Keys and local directories

# Set the Groq API key
os.environ['GROQ_API_KEY'] = groq_token 

# Specify the folder path
folder_path = folder_path

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get('GROQ_API_KEY'),
)

# load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large', legacy = False)
device = torch.device('cpu')

# Dictionary to hold the key points for each text file
key_points_dict = {}

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            preprocessed_text = content.strip().replace('\n', '')
            input_text = 'summarize: ' + preprocessed_text
            tokenized_text = tokenizer.encode(input_text, return_tensors = 'pt', truncation = True, max_length = 512). to(device)
            summary_ids = model.generate(tokenized_text, min_length = 40, max_length = 150, length_penalty = 2.0, num_beams = 4, early_stopping = True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)

            print(f'Summary for {filename}: \n{summary}\n')

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Provide a bulleted list of key points: {content}",
                    }
                ],
                model="llama3-8b-8192",
            )
            # Get the bulleted list of key points
            bulleted_content = chat_completion.choices[0].message.content
            
            # Save the key points in the dictionary
            key_points_dict[filename] = bulleted_content
            
# Save the key points dictionary to a JSON file in the same directory
json_file_path = os.path.join(folder_path, 'key_points.json')
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(key_points_dict, json_file, ensure_ascii=False, indent=4)