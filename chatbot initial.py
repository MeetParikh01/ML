import nltk
import numpy as np
import random
import string
# #f = open('/home/meet/ML/chatbot.txt', 'rb+')
# raw=f.read()
# raw=raw.lower()
# #nltk.download('punkt')
# #nltk.download('wordnet')
# sent_tokens = nltk.sent_tokenize(raw)
# word_tokens = nltk.word_tokenize(raw)
# print('send_tokens=',send_tokens,'word_tokens=',word_tokens )
print('hi')
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize

#EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
f = open("chatbot.txt", "r")
for word in f:
	print(sent_tokenize(word))
fileread=f.read()
f.close()
# for line in fileread:
# 	for word in line.split():
# 		print(word)
#for word in fileread:
#	print(sent_tokenize(word))
#print(sent_tokenize(fileread))
#print(sent_tokenize(fileread))
