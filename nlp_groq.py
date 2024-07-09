import os
import json

# NLP Dependencies Import
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer

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

# Dictionary to hold the key points for each text file
key_points_dict = {}

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a text file
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        # Open and read the file with utf-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Use Groq API to summarize the content
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize the following text to an 8th grade level and provide a bulleted list of key points: {content}",
                    }
                ],
                model="llama3-8b-8192",
            )

            # Get the summarized content
            summarized_content = chat_completion.choices[0].message.content
            
            # Save the key points in the dictionary
            key_points_dict[filename] = summarized_content

# Save the key points dictionary to a JSON file in the same directory
json_file_path = os.path.join(folder_path, 'key_points.json')
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(key_points_dict, json_file, ensure_ascii=False, indent=4)
