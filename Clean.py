import pandas as pd
import getTickers

valid_tickers = getTickers.get_sp500_tickers()

def clean_tickers(ticker_list):
    if isinstance(ticker_list, str):
        tickers = ticker_list.split(",")  # Split the string by commas
        tickers = [ticker.strip() for ticker in tickers]  # Clean any extra spaces
        valid_tickers_in_list = [ticker for ticker in tickers if ticker in valid_tickers]
        if valid_tickers_in_list:
            return valid_tickers_in_list
    return None

def clean_content(content, title):
    if isinstance(content, str):
        return content.replace("\n", " ").replace("\r", " ").strip()
    elif pd.isna(content):
        return title
    else:
        return str(content).strip()

def clean_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    df = df[~((df['title'].str.contains("update", case=False, na=False)) & df['content'].isna())]
    df['content'] = df.apply(lambda row: clean_content(row['content'], row['title']), axis=1)
    df['tickers'] = df['tickers'].apply(clean_tickers)

    df = df.dropna(subset=['tickers'])
    df_grouped = df.groupby('title').agg({
        'id': 'first',
        'score': 'first',
        'content': ' '.join,
        'url': 'first',
        'subreddit': 'first',
        'tickers': lambda x: ' '.join([str(ticker) for ticker in x if isinstance(ticker, str)]),  # Join tickers
    }).reset_index()
    df_grouped = df[['id', 'title', 'score', 'content', 'url', 'subreddit', 'tickers']]
    df_grouped.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/raw_data/reddit_finance_posts.csv"  # Adjust the file path if necessary
    output_file = "data/processed_data/reddit_finance_posts_cleaned.csv"  # Output cleaned file
    clean_csv(input_file, output_file)
