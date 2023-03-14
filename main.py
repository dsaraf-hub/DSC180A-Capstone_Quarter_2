"""
Wisdom of the Crowd
https://en.wikipedia.org/wiki/Wisdom_of_the_crowd
"""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import pandas as pd
import openai
import os
import re
from helper_functions import is_word_present, get_embedding, extract_tickers

os.environ["OPENAI_API_KEY"] = #YOUR OPENAI KEY HERE
openai.api_key = #YOUR OPENAI KEY HERE

df = pd.read_csv('stock_tweets.csv')
df = df['Tweet'].dropna()

# list of words to be searched
words_to_search = ['TSLA', 'GOOGL', 'TWTR', 'META', 'AMZN']

# applying the function to each string in the series
result = df.apply(lambda x: is_word_present(x, words_to_search))
df = df.loc[result]

cleaned_embeds = df.copy().to_frame()
cleaned_embeds['$_count'] = cleaned_embeds['Tweet'].apply(lambda x: x.count('$'))
cleaned_embeds = cleaned_embeds.loc[cleaned_embeds['$_count'] <= 3]

vals = cleaned_embeds['Tweet'].values.tolist()
raw_embeddings = []
for i in range(0, len(cleaned_embeds), 10):
  lower = i
  upper = lower + 10
  treat = vals[lower:upper]
  raw_embeddings.extend(get_embedding(treat))

cleaned_embeds = cleaned_embeds.reset_index(drop = True)
cleaned_embeds['embeddings'] = pd.Series(raw_embeddings)
cleaned_embeds['tickers'] = extract_tickers(cleaned_embeds['Tweet'], words_to_search)






