import pandas as pd

data = pd.read_csv('data/processed_data/reddit_finance_posts_cleaned.csv')
def generate_title_content_prompt(row):
    return f"""
    Analyze the following financial Reddit post for sentiment.

    - **ID**: "{row['id']}"
    - **Title**: "{row['title']}"
    - **Content**: "{row['content']}"
    - **Subreddit**: "{row['subreddit']}"
    - **Mentioned Tickers**: {row['tickers']}

    Tasks:
    1. Classify the **sentiment of the title** as Positive, Negative, or Neutral.
    2. Classify the **sentiment of the content** as Positive, Negative, or Neutral.
    3. Provide reasons for the title sentiment classification.
    4. Provide reasons for the content sentiment classification.

    Output the results in this structured format:
    {{
      "title_sentiment": "...",
      "title_reasons": "...",
      "content_sentiment": "...",
      "content_reasons": "..."
    }}
    """

data['prompts'] = data.apply(generate_title_content_prompt, axis=1)

data['prompts'].to_csv('data/processed_data/title_content_prompts.csv', index=False)

