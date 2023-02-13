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
import pinecone
import helper_functions

os.environ["OPENAI_API_KEY"] = #YOUR OPENAI KEY HERE
openai.api_key = #YOUR OPENAI KEY HERE

api_key = #YOUR PINECONE API KEY HERE
pinecone.init(api_key=api_key, environment='us-west1-gcp')


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

vectors_to_upsert = []
ctr = 0
for i in cleaned_embeds['embeddings']:
  temp = []
  temp.append(str(ctr))
  temp.append(i[1])
  vectors_to_upsert.append(tuple(temp))
  ctr = ctr + 1

with pinecone.Index('categories', pool_threads=30) as index:
    # Send requests in parallel
    for i in range(0, len(vectors_to_upsert), 300):
      async_results = [
          index.upsert(vectors=chunk, async_req=True)
          for chunk in chunks(vectors_to_upsert[i:i+300])
      ]
      # Wait for and retrieve responses (this raises in case of error)
      [async_result.get() for async_result in async_results]
      print(i)

"""#Next Steps


*   Label a subset of these tweets with bullish, bearish, neutral
*   Train a classifier to label the rest of the tweets
*   Based on classification, group the labels by each ticker and give each ticker a buy/sell/hold signal
* We would do this for a day in advance for a few days and see if this works.
* If time permits, we'd also like to try this on small or medium cap stocks. However, limitations with this are possibly not enough tweets. 


"""

