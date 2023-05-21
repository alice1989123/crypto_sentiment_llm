
def count_sentiment (tweets : list):
    
    positive = 0
    negative = 0
    neutral = 0
    for tweet in tweets:
        try:
            if  'Bullish'in tweet:
                positive += 1
            elif 'Bearish'in tweet:
                negative += 1
            else:
                neutral += 1
        except Exception as e:
            pass
    return {"Bullish":positive, "Bearish":negative, "Neutral":neutral}
