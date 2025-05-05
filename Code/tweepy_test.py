import tweepy

# Replace these values with your credentials
consumer_key = 'Oyg7HUX57WfrRsutTxhAMok8C'
consumer_secret = 'XdKCTS2gVPPjtCG2WysbgZkJGfuWa0h5HsOLf6G5AczC5gaglV'
access_token = '1871759832262615040-4PfbMyC199QbeORq5lu91bOihXxZnB'
access_token_secret = 'WGgMfXv6T88a7mkqfVsNCixXyBvP8lD04F1FEgNsEGrWW'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAEjK0AEAAAAAxaGBvLG4AYzou9g%2F5Rwlr8ZrnH4%3D8zgl7YbgtfSfNGzSBggoa8m9xPtJJ2Ti0XwHNfSiynXKzfN5Yk'


# Set up OAuth1 authentication
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Initialize client
client = tweepy.Client(bearer_token=bearer_token)

# Replace 'tweet_id' with the actual tweet ID you want to retrieve
tweet_id = '1876389361862008985'

# Create a query to search for replies to the tweet
query = f"conversation_id:{tweet_id} -from:{client.get_tweet(tweet_id).data['author_id']}"

# Search for recent replies
try:
    response = client.search_recent_tweets(query=query, tweet_fields=["author_id", "created_at", "text"])
    
    print(f"Replies to Tweet ID {tweet_id}:\n")
    for tweet in response.data:
        print(f"User ID: {tweet.author_id} - Created at: {tweet.created_at}\nTweet: {tweet.text}\n")
except Exception as e:
    print(f"Error: {e}")




 
