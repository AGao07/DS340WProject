import pandas as pd
import getTickers

valid_tickers = getTickers.get_sp500_tickers()

def clean_tickers(ticker_list):
    if isinstance(ticker_list, str):
        tickers = ticker_list.split(",")
        tickers = [ticker.strip() for ticker in tickers]
        valid_tickers_in_list = [ticker for ticker in tickers if ticker in valid_tickers]
        if valid_tickers_in_list:
            return valid_tickers_in_list
    return None


def clean_content(content):
    return content.replace("\n", " ").replace("\r", " ").strip()


def clean_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    df['content'] = df['content'].fillna("").apply(clean_content)
    df['tickers'] = df['tickers'].apply(clean_tickers)
    df = df.dropna(subset=['tickers'])
    if 'tickers' in df.columns and df['tickers'].isnull().all():
        df = df.drop(columns=['tickers'])
    df.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/raw_data/reddit_finance_posts.csv"  # Adjust the file path if necessary
    output_file = "data/processed_data/reddit_finance_posts_cleaned.csv"  # The output cleaned file
    clean_csv(input_file, output_file)