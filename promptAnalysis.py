import openai
import pandas as pd

openai.api_key = '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'

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
