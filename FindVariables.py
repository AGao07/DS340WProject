import pandas as pd
import numpy as np
import re

df = pd.read_csv("data/processed_data/Disagreements.csv")

def count_numbers(text):
    return len(re.findall(r'\d', text))

def count_special_characters(text):
    return len(re.findall(r'[^A-Za-z0-9\s]', text))

def count_words(text):
    return len(text.split())

def average_word_length(text):
    words = text.split()
    if len(words) == 0:
        return 0
    return np.mean([len(word) for word in words])

df['title_length'] = df['original_title'].apply(len)
df['content_length'] = df['original_content'].apply(len)

df['title_number_count'] = df['original_title'].apply(count_numbers)
df['content_number_count'] = df['original_content'].apply(count_numbers)

df['title_special_char_count'] = df['original_title'].apply(count_special_characters)
df['content_special_char_count'] = df['original_content'].apply(count_special_characters)

df['title_word_count'] = df['original_title'].apply(count_words)
df['content_word_count'] = df['original_content'].apply(count_words)

df['title_avg_word_length'] = df['original_title'].apply(average_word_length)
df['content_avg_word_length'] = df['original_content'].apply(average_word_length)

df.to_csv("data/processed_data/Enhanced_Disagreements.csv", index=False)
