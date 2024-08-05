import os
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import Counter
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from authtoken import folder_path  # folder_path is unique to your local directory

# Download NLTK data if not already available
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

# Load models and tokenizers
t5_model = T5ForConditionalGeneration.from_pretrained('t5-large')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-large', legacy=False)
device = torch.device('cpu')

# Dictionary to hold the summaries and descriptions for each text file
summaries_dict = {}
description_dict = {}

# Specify the folder path
folder_path = folder_path

# Define stop words and prepositions to exclude
stop_words = set(stopwords.words('english'))
prepositions = set([
    'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'under', 'over', 'during', 'before', 'after'
])

def generate_summaries(text, model, tokenizer, device):
    input_text = 'summarize: ' + text
    tokenized_text = tokenizer.encode_plus(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    summary_ids = model.generate(tokenized_text['input_ids'], min_length=40, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_description(summary):
    # Tokenize and tag parts of speech
    words = word_tokenize(summary)
    tagged_words = pos_tag(words)
    
    # Filter out non-noun words and prepositions
    nouns = [word for word, tag in tagged_words if tag.startswith('NN') and word.lower() not in stop_words and word.lower() not in prepositions]
    
    # Get the three most common nouns
    noun_freq = Counter(nouns)
    most_common_nouns = [word for word, freq in noun_freq.most_common(3)]
    
    # Ensure the output is exactly three nouns
    if len(most_common_nouns) < 3:
        most_common_nouns.extend(['fallback', 'words', 'here'])  # Fallback nouns if not enough common nouns
    return ' '.join(most_common_nouns)

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            preprocessed_text = content.strip().replace('\n', ' ')

            # Summary Function Call
            summary = generate_summaries(preprocessed_text, t5_model, t5_tokenizer, device)
            summaries_dict[filename] = summary

            # Description Function Call
            description = generate_description(summary)
            description_dict[filename] = description

            print(f'Summary for {filename}: \n{summary}\n')
            print(f'Description for {filename}: \n{description}\n')

# Save the summaries and descriptions dictionary to a JSON file in the same directory
summaries_json_file_path = os.path.join(folder_path, 'summaries.json')
with open(summaries_json_file_path, 'w', encoding='utf-8') as summaries_json_file:
    json.dump(summaries_dict, summaries_json_file, ensure_ascii=False, indent=4)

descriptions_json_file_path = os.path.join(folder_path, 'descriptions.json')
with open(descriptions_json_file_path, 'w', encoding='utf-8') as descriptions_json_file:
    json.dump(description_dict, descriptions_json_file, ensure_ascii=False, indent=4)