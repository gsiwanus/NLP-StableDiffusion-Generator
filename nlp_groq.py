import os
import json

# NLP Dependencies Import
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# GROQ AI Import
from groq import Groq
from authtoken import groq_token, folder_path # authtoken and folder_path are unique to your API Keys and local directories

# Set the Groq API key
os.environ['GROQ_API_KEY'] = groq_token 
# Specify the folder path
folder_path = folder_path

# Initialize NLTK stopwords and tokenizer
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Summarization Function
def nltk_summarize(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]

    freq_dist = FreqDist(words)
    most_freq_words = freq_dist.most_common(10)
    summary = [word[0] for word in most_freq_words]

    return ' '.join(summary)

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

            # Summarize articles using NLTK NLP
            summarized_content = nltk_summarize(content)

            # Use Groq API to identify key points
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Create a bulleted list of key points based on the summarized content: {summarized_content}",
                    }
                ],
                model="llama3-8b-8192",
            )

            # Get the summarized content
            key_points = chat_completion.choices[0].message.content
            
            # Save the key points in the dictionary
            key_points_dict[filename] = key_points

# Save the key points dictionary to a JSON file in the same directory
json_file_path = os.path.join(folder_path, 'key_points.json')
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(key_points_dict, json_file, ensure_ascii=False, indent=4)
