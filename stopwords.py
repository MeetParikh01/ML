import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#print(set(stopwords.words('english')))
stopwords=list(stopwords.words('english'))
file_stopwords=[]
words=[]

# with open('exp.txt','a+') as openfile:
# 	openfile.write('naoojasnoasboncsaobo')


with open("chatbot.txt") as openfile:
	for line in openfile:
		for part in line.split():
			words.append(part)

useful_words=[word for word in words if word not in stopwords]
# print(useful_words)
for word in words:
	if word in stopwords:
		file_stopwords.append(word)
print(list(set(file_stopwords)))