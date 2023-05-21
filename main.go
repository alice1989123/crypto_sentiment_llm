
package main

import (
    "context"
    "fmt"
    "time"
    twitterscraper "github.com/n0madic/twitter-scraper"
)

func main() {
    scraper := twitterscraper.New()
    scraper.Login("Alicia_Basilio0", "pollo3.1416bill")

    // Get the current date and the date one day ago
    today := time.Now().Format("2006-01-02")
    scraper.WithDelay(5)
    scraper.WithReplies(true)
    yesterday := time.Now().AddDate(0, 0, -1).Format("2006-01-02")

    // Construct the query string
    query := fmt.Sprintf("bitcoin news since:%s until:%s -filter:retweets", yesterday, today)

    for tweet := range scraper.SearchTweets(context.Background(), query, 1000) {
        if tweet.Error != nil {
            panic(tweet.Error)
        }
        fmt.Println(tweet.Text)
    }
}