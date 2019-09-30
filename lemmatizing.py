from nltk.stem import WordNetLemmatizer
import nltk
lemmatizer=WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("rocks",pos='r'))
print(dir(nltk))