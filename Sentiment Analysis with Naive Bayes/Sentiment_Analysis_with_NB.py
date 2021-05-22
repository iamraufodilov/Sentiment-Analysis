#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# INTRODUCTION
# We help simplify sentiment analysis using Python in this tutorial. 
# You will learn how to build your own sentiment analysis classifier using Python and 
# understand the basics of NLP (natural language processing).

# loading necessary libraries
import nltk
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Define the feature extractor

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
#_>print(nltk.classify.accuracy(classifier, test_set))# that is good our model evaluate test dataset with 74% accuracy

# Show the most important features as interpreted by Naive Bayes
#_>classifier.show_most_informative_features(5)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION

'''
This is small sentiment analysis model for classifying text with the help of NLTK
As classifier we use Naive Bayes which is common NLP classifier 
'''


