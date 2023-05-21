from btc_scrapper import btc_scrapper
import logging as log
from process_tweets import process_tweets
import pickle
from Utils.Util import count_sentiment
from datetime import datetime as dt
from mongo import saveMongo
import time
log.basicConfig(filename='btc_scrapper.log', level=log.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')


def main():
    now_str = dt.now().strftime("%Y-%m-%d %H:%M:%S")


    try:
        btc_scrapper(now_str)
        log.info("Done scrapping ...")
        print("Done scrapping ...")
        
    except Exception as e:
        log.exception("Exception occurred scrapping" )
        print("Exception occurred scrapping")
    print("Processing tweets ...")
    tweets = process_tweets(now_str)
    log.info("Done processing tweets ...")
    print("Done processing tweets ...")
    # Use a context manager to open a file in write binary mode ('wb')
    print( tweets)
    sentiment = count_sentiment(tweets["tweets_resumes"])
    sentiment["count"]= tweets["tweets_number"]
    # Get current UTC time
    saveMongo(filter={"date":now_str} ,data = sentiment , collection="sentiment" ) 
    saveMongo(data = {"news": tweets["tweets_resumes"] }, collection = "news" , filter={"date":now_str}) 




if __name__ == '__main__':
    while True:
        main()
        time.sleep(24 * 60 * 60)  # 24 hours in seconds



