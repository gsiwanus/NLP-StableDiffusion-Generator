import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from authtoken import folder_path # folder_path is unique to your local directory

# load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large', legacy = False)
device = torch.device('cpu')

# Dictionary to hold the summaries and key points for each text file
summaries_dict = {}
key_points_dict = {}
description_dict = {}

# Specify the folder path
folder_path = folder_path

def generate_summaries(text, model, tokenizer, device):
    input_text = 'summarize: ' + text
    tokenized_text = tokenizer.encode_plus(input_text, return_tensors = 'pt', truncation = True, max_length = 512).to(device)
    summary_ids = model.generate(tokenized_text['input_ids'], min_length = 40, max_length = 150, length_penalty = 2.0, num_beams = 4, early_stopping = True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    return summary

def generate_key_points(text, model, tokenizer, device):
    input_text = 'summarize: ' + text + ' in bullet points'
    tokenized_text = tokenizer.encode_plus(input_text, return_tensors = 'pt', truncation = True, max_length = 512).to(device)
    summary_ids = model.generate(tokenized_text['input_ids'], min_length = 40, max_length = 150, length_penalty = 2.0, num_beams = 4, early_stopping = True)
    key_points = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    return key_points

def generate_description(text, model, tokenizer, device):
    input_text = 'summarize: ' + text + ' in three words'
    tokenized_text = tokenizer.encode_plus(input_text, return_tensors= 'pt', truncation = True, max_length = 512).to(device)
    summary_ids = model.generate(tokenized_text['input_ids'], min_length = 40, max_length = 150, length_penalty = 2.0, num_beams = 4, early_stopping = True)
    description = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    return description 

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            preprocessed_text = content.strip().replace('\n', '')

            ''' input_text = 'summarize: ' + preprocessed_text
            tokenized_text = tokenizer.encode_plus(input_text, return_tensors = 'pt', truncation = True, max_length = 512). to(device)
            summary_ids = model.generate(tokenized_text['input_ids'], min_length = 3, max_length = 10, length_penalty = 2.0, num_beams = 4, early_stopping = True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)

            print(f'Summary for {filename}: \n{summary}\n') '''

            # Summary Function Call
            summary = generate_summaries(preprocessed_text, model, tokenizer, device)
            summaries_dict[filename] = summary

            # Key Points Function Call
            key_points = generate_key_points(preprocessed_text, model, tokenizer, device)
            key_points_dict[filename] = key_points

            # Description Function Call
            description = generate_description(preprocessed_text, model, tokenizer, device)
            description_dict[filename] = description

            print(f'Summary for {filename}: \n{summary}\n')
            print(f'Key Points for {filename}: \n{key_points}\n')
            print(f'Description for {filename}: \n{description}\n')

# Save the summaries and key points dictionary to a JSON file in the same directory
json_file_path = os.path.join(folder_path, 'key_points.json')
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(key_points_dict, json_file, ensure_ascii=False, indent=4)

descriptions_json_file_path = os.path.join(folder_path, 'descriptions.json')
with open(descriptions_json_file_path, 'w', encoding = 'utf-8') as descriptions_json_file:
    json.dump(description_dict, descriptions_json_file, ensure_ascii = False, indent = 4)