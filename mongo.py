import datetime
from typing import List, Dict
from pymongo import MongoClient
import time
import os 
DB_KEY =  os.getenv("DB_KEY") 

def saveMongo(data: List[Dict[str, str]], filter:Dict  , collection : str):
    # Configure your MongoDB connection
    connection_string = DB_KEY
    client = MongoClient(connection_string)

    # Select the database and collection
    db = client['crypto_tweets_resume']
    general_collection = db[collection]

   
    # Save the predictions to the collection under a single key
    general_collection.update_one(
        filter,
        {'$set': data},
        upsert=True
    )