import pandas as pd
import os 
from langchain.embeddings import OpenAIEmbeddings
from sklearn.cluster import KMeans
import re
import string
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from langchain.chat_models import ChatOpenAI
import logging as log
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv
# Load the .env file. By default, it looks for a .env file in the same directory as the script
load_dotenv()


import logging as log
log.basicConfig(filename='process_tweets.log', level=log.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def format_tweets(tweet_list):
    return ', '.join([f' :-> {tweet}' for tweet in tweet_list])


def preprocess_text(text):
    tweets = pd.read_csv('btc.csv')
    tweets = tweets.iloc[:-1 ,-1 ]    
    # Remove URLs
    text = re.sub(r"http\S|www\S|https\S", '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase to maintain consistency
    text = text.lower()

    # Split the text into words
    words = text.split()
    
    # Remove tweets with less than 5 words
    if len(words) < 5:
        return None
    
    return text

def resume_tweeets(cluseter_tweets):
    try:
        formatted_tweets = format_tweets(cluseter_tweets)

        chat = ChatOpenAI(temperature=1 ,openai_api_key=OPENAI_API_KEY , model_name="gpt-3.5-turbo")
        messages = [
            SystemMessage(content="Please  sumarize as much as posible the idea, highlight any important information  focus only in cryptocurrency, ignore spam, if is nothing important, or cant give detailed info just write 'nothing important' the finnal output should start with 'Bullish: ' 'Bearish: ' or 'Neutral: '  depending of how the news afect cryptocurrency"),
            HumanMessage(content=formatted_tweets)
        ]
        response = chat(messages).content
        print("News:" , response)
        if ("nothing important" in response.lower()):
            pass
        else:
            return response
    except Exception as e:
        log.error(e)

def process_tweets(now_str):
    try:
        tweets = pd.read_csv('btc'+now_str+'.csv')
        tweets = tweets.iloc[: ,-1 ]    
        # Apply preprocess_text function to tweets
        tweets = tweets.apply(preprocess_text)
        tweets = tweets.dropna()  # Remove None values from the DataFrame
    except Exception as e:
        log.error("Error processing tweets")

    tweets_list= [ str(tweet) for tweet in tweets]

    vectors = []
    for tweet in tweets_list:
        try:
            embedingTweet = embeddings.embed_query(tweet)  
            vectors.append(embedingTweet)
        except Exception as e:
            log.error("Error connecting to OpenAi tweet")

    try:
        matrix = np.vstack(vectors)
        n_clusters = int(len(tweets_list)/10)
        kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
        kmeans.fit(matrix)
        df = pd.DataFrame({'tweet content': tweets_list})
        df["embeding"]=matrix.tolist()
        df['cluster'] = kmeans.labels_
        # Calculate the closest points to the cluster centers
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, matrix)
        # Find the top tweets for each cluster
        top_tweets = df.loc[closest].sort_values(by='cluster')
        # If you want to get more than one tweet per cluster, you can use the following:
        top_n = 10  # Change this to get more or less tweets per cluster
        top_tweets_each_cluster = {}

        for cluster_id in range(n_clusters):
            # Find all tweets in this cluster
            cluster_tweets = df[df['cluster'] == cluster_id]
            
            # Calculate the distance of all tweets in this cluster to the cluster center
            distances = np.linalg.norm(cluster_tweets['embeding'].tolist() - kmeans.cluster_centers_[cluster_id], axis=1)
            
            # Find the indices of the tweets with the smallest distances
            closest_indices = distances.argsort()[:top_n]
            
            # Add the top tweets for this cluster to the dictionary
            top_tweets_each_cluster[cluster_id] = cluster_tweets.iloc[closest_indices]['tweet content'].tolist()
    except Exception as e:
        log.error("Error clustering tweets")
    tweets_resumes = []
    for i  in range(n_clusters):
        try:
            resume = resume_tweeets( top_tweets_each_cluster[i])
            if resume is not None:
                tweets_resumes.append(resume)
        except Exception as e:
            log.error("Error processing tweets")
    return { "tweets_resumes":tweets_resumes , "tweets_number": len(tweets_list)  }
