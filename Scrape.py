import praw
import pandas as pd
import re

reddit = praw.Reddit(
    client_id='VrIL6ECM-p_kqqynXK6Ucw',
    client_secret='C-ICFTty213-mlbxk-5zELH1LhtaLQ',
    user_agent='Education_NLP by /u/TakafumiKusonori'
)

# File to store post IDs that have already been processed
processed_file = "processed_posts.txt"

# Load existing processed post IDs
try:
    with open(processed_file, "r") as file:
        processed_ids = set(line.strip() for line in file)
except FileNotFoundError:
    processed_ids = set()

# Specify subreddits and settings
subreddits = "stocks+wallstreetbets+investing"
subreddit = reddit.subreddit(subreddits)

# Define regex for ticker symbols (assumes tickers are uppercase and 1-5 characters)
ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

# List to store new posts
new_posts = []

# Fetch new posts
for post in subreddit.new(limit=100):  # Fetch up to 100 posts
    if post.id not in processed_ids:
        # Extract tickers from title and content
        tickers_in_title = ticker_pattern.findall(post.title)
        tickers_in_content = ticker_pattern.findall(post.selftext)
        tickers = list(set(tickers_in_title + tickers_in_content))  # Combine and remove duplicates
        post_content_cleaned = post.selftext.replace("\n", " ")

        # Add post details to the list
        new_posts.append({
            "id": post.id,
            "title": post.title,
            "score": post.score,
            "content": post.selftext,
            "url": post.url,
            "subreddit": post.subreddit.display_name,
            "tickers": ", ".join(tickers)  # Save tickers as a comma-separated string
        })

        # Mark post as processed
        processed_ids.add(post.id)

# Save new post IDs to the file
with open(processed_file, "a") as file:
    for post in new_posts:
        file.write(post["id"] + "\n")

# Check if new_posts has data
if new_posts:
    # Create a DataFrame from the list of new posts
    df = pd.DataFrame(new_posts)

    # Check the DataFrame content
    print("DataFrame created successfully:")
    print(df)

    # Append to the CSV file (or create if it doesn't exist)
    df.to_csv("data/raw_data/reddit_finance_posts.csv", mode='a', header=False, index=False)
else:
    print("No new posts to save.")