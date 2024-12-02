Parent Papers

FinBERT.

FinTRUST.

PyFin-Sentiment.

Analysis on social media text analysis.

pip install -r requirements.txt


############# file instruction #############

Running Clean.py
will only involve posts that specifically mention a stock, and make clean the formatting.
Each run can either be sorted by "relevance", "hot", "top", "new", or "comments"
There is a running count of downloaded post ids in processed_posts.txt, and it is a large aggregate of posts.
After running both functions, there will be "reddit_finance_posts.csv" in data/raw_data and "reddit_finance_posts_cleaned.csv" in data/processed_data.


The files under Test folder use the demo from parent paper, it was to test if our cleaned data is accepted by the model or not.

Scrape.py 
gets 100 posts (Rate limit by 100 posts for each run and limited to 60 runs per hour) from r/investing, r/stocks, and r/wallstreetbets. It is first generation.

ScrapeScheduler.py 
fetches up to 6000 posts per run, validates extracted stock tickers against a predefined list, also runs every hour using a scheduler for continuous data collection. Unlike the first file, it includes error handling, logging, and processes only posts containing valid stock tickers for more reliable data.

annotationAssistant.py
This file is a manual annotation of the CSV file of Reddit post data. Read the input CSV and check if the output file already exists. If the output CSV does not exist, create one, then iterate through the posts that have not yet been annotated and manually select the sentiment annotation (" bullish, "" neutral," or "bearish") in the input field.

testaccuracy.py
This document performs accuracy analysis of the sentiment classification of the two models (compared to the other from the gpt model), merges the two data sets, calculates baseline accuracy measures, and measures the consistency of each model under different transformations.

union.ipynb under Final folder 
This is a syndication file, which is a workflow for analyzing the differences in sentiment between the different models applied to Reddit finance posts. The process starts with sentiment analysis using two models, FinBERT and PyFin, to classify the titles and content of financial posts. Check the consistency of the predictions of these models under different transformations, including negation, reordering, and transitivity. This includes identifying the divergence between the two models and enhancing the divergence dataset with additional linguistic features such as word counts, special character counts, and average word length. Finally, a logistic regression model was built to understand the relationship between these characteristics and the observed disagreement, providing insight into factors that may influence the prediction of divergent sentiment. It aims to enhance understanding of emotional predictive behavior and the inconsistencies of multiple models on financial content.


