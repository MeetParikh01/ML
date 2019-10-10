from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

ckey = ''
csecret = ''
atoken = ''
asecret = ''

class Listener(StreamListener):

	def on_data(self, data):
		all_data = json.loads(data)
		tweet = all_data["text"]
		sentiment_value, confidence = s.sentiment(tweet)
		print(tweet, sentiment_value, confidence)

		
		output = open("twitter-out.txt","a")
		output.write(tweet +'\t' + sentiment_value + '\t' + str(confidence))
		output.write('\n')
		output.close()
		return True

	def on_error(self, status):
		print(status)
		return False

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track = ['Trump'])

