import nltk
from nltk.corpus import wordnet

# print(dir(wordnet))
syns=wordnet.synsets('program')
print(syns)

x=syns[0].lemmas()
print('++++++++++++++++++++++++++')
for i in x:
	print(i.name())
print('==========================')
for i in syns:
	print(i.definition())

print('4658461',syns[0].definition())	

print('===== examples ====')
 
for i in syns:
	print('####################')
	print(i,i.examples())

print('antonyms and synonyms')

synonyms,antonyms=[],[]

for syn in wordnet.synsets('good'):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

print(wordnet.synset('ship.n.01').wup_similarity(wordnet.synset('boat.n.01')))
