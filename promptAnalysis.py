import openai
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OpenaiAPI")

data = pd.read_csv('data/processed_data/title_content_prompts.csv')

def analyze_sentiment(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Apply GPT analysis
data['analysis'] = data['prompts'].apply(analyze_sentiment)

# Save the results
data.to_csv('data/processed_data/sentiment_analysis_results.csv', index=False)
