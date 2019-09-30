import nltk,tkinter
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import treebank_chunk
#nltk.download('words')
train_text = state_union.raw("mhp.txt")
print(train_text)

sample_text = state_union.raw("chp.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
print(custom_sent_tokenizer)

tokenized = custom_sent_tokenizer.tokenize(sample_text)
print(tokenized)
print('The nltk version is {}.'.format(nltk.__version__))

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)	
            #print(words)
            tagged = nltk.pos_tag(words)
            #print(tagged)
            #chunkGram = r"""Chunk: {<PRB.?>*<VB.?>*<PRP>+<VBP>?}"""
            #chunkGram = r"""pointer: {<VB>+<RP>|<NNP>+<VBP>+<PRP> |(<NNP>*)| <PRP>+<VBP>+<JJ> }"""
            chunkGram = r"""pointer: {<VB>+<RP>} 
            				abc:{(<NNP>|<PRP>)+<VBP>+(<VB>+<PRP.><NN>|<PRP>)}
             				"""
            # chunkGram = r"""pointer: {<VB>+<RP>} 
            # 				abc:{(<NNP>|<PRP>)+<VBP><VB><PRP.?>}
            # 				"""
            chunkParser = nltk.RegexpParser(chunkGram)
            #print(chunkParser)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            # print('=============== x ===============')
            # x=str(chunked)
            # y=[]
            # for sen in chunked:
            # 	print('+++++',isinstance(sen, nltk.tree.Tree))
            # 	if isinstance(sen, nltk.tree.Tree) and sen.label() == 'NNP':
            # 	 	print(sen.label)
            # 	 	noun_phrase = ""
            # 	 	for (org, tag) in sen.leaves():
            # 	 		noun_phrase += org + ' '
            # 	 	y.append(noun_phrase.rstrip())

            # print(y,'qwertyui')


            # print(x,'whqownouwou')
            
            chunked.draw()
    except Exception as e:
        print(str(e))

process_content()