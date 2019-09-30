import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
words=[]

with open("exp.txt") as openfile:
	for line in openfile:
		for part in line.split():
			words.append(part)
stem_words=[]
for word in words:
	stem_words.append(ps.stem(word))
	#print(ps.stem(word))

print(stem_words)