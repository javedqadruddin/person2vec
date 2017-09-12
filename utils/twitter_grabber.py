import sys
import tweepy


CONSUMER_KEY = ''
CONSUMER_SECRET = ''

ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''

with open('../etc/twitter.cfg', 'r') as f:
    CONSUMER_KEY = f.next().rstrip()
    CONSUMER_SECRET = f.next().rstrip()
    ACCESS_TOKEN = f.next().rstrip()
    ACCESS_TOKEN_SECRET = f.next().rstrip()


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

public_tweets = api.user_timeline(id="realDonaldTrump", count=100, page=sys.argv[1])

with open('trump_test.txt', 'w') as f:
    for tweet in public_tweets:
        print(tweet.text)
        f.write(u' '.join(tweet.text).encode('utf-8'))
