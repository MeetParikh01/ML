import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw("mhp.txt")
print(train_text)

sample_text = state_union.raw("chp.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
print(custom_sent_tokenizer)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			namedEnt = nltk.ne_chunk(tagged, binary=False)
			namedEnt.draw()
	except Exception as e:
		print(str(e))

process_content()
