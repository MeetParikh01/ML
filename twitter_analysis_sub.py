import nltk,random,pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier.classify(features)
            votes.append(vote)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("positive.txt","r", encoding="ISO-8859-1").read()
short_neg = open("negative.txt","r", encoding="ISO-8859-1").read()

documents = []

for review in short_pos.split('\n'):
    documents.append( (review, "pos") )

for review in short_neg.split('\n'):
    documents.append( (review, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for word in short_pos_words:
    all_words.append(word.lower())

for word in short_neg_words:
    all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# positive data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

# classifier_open = open('twitter.pickle', 'rb')
# classifier = pickle.load(classifier_open)class VoteClassifier(ClassifierI):
   
# classifier_open.close()

# print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",(nltk.classify.accuracy(BNB_classifier, testing_set))*100)