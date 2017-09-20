#!/usr/bin/python

'''
    @author Ryan Milbourne
    @date   September 4, 2017
    @brief  Main driver for the project.  Creates a sentiment classifier, then trains a spam/ham
            classifier using sentiment as one of the the features.
            One can also enable or disable tf-idf as a feature for the main classifier.
'''

import sys
# Python shenanigans
sys.path.append('/usr/local/lib/python2.7/site-packages')

# External imports
from sklearn import datasets
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cross_validation import KFold
import numpy
import time

# Internal imports
from processData import processData
from processData import readSentiment
from features import CharacterCountTransform
from features import DollarSignCountTransform
from features import NumericTokenTransform
from features import AlphaTokenTransform

# Helpful constants
SPAM = 'spam'
HAM = 'ham'

DATAFILE_PATH = './dataSet/SMSSpamCollection.txt' # Relative path to file
POS_SENTIMENT_PATH = './rt-polaritydata/rt-polarity.pos' # Relative path to file
NEG_SENTIMENT_PATH = './rt-polaritydata/rt-polarity.neg' # Relative path to file
SENTIMENT_PATH = './rt-polaritydata/rt-polarity' # Relative path to file


################
# Fetch the data
################
print("=======================")
print("Reading in SMS data...")
print("=======================")
start = time.time()

dataz = processData(DATAFILE_PATH)

end = time.time()
duration = end-start

# Some error checking
numTexts = len(dataz['text'].values)
numLabels = len(dataz['label'].values)
if numTexts != numLabels:
    print("unknown error: text/label mismatch")
    exit(1)

print("Read in %d SMS messages") % (numTexts)
print("Data read took %.2fsec\n") % (duration)

print "Hold on, we gotta make a sentiment analyzer first"
sentiLabels = ['pos','neg'] # Initially we'll start off with binary sentiment
start = time.time()
sentiData = readSentiment(SENTIMENT_PATH)
end = time.time()
duration = end-start
print("Data read took %.2fsec\n") % (duration)

################
# Extract all features
################
print("=======================")
print("Extracting features...")
print("=======================")

print ("Training sentiment analyzer...")
sentiPipe = Pipeline([
        ('tf_idf_vec', TfidfVectorizer(stop_words='english',min_df=10,max_df = 0.8, sublinear_tf=True, use_idf=True)),
        ('classifier', MultinomialNB())                    # The classifier itself
])

sentiFold = KFold(n=len(sentiData['text']), n_folds=10)
sentiScores = []
sentfusionMatrix = numpy.array([[0,0], [0,0]])

for trainingIdxs, testingIdxs in sentiFold:
    # Data to train with
    trainingText = sentiData.iloc[trainingIdxs]['text'].values
    trainingLabels = sentiData.iloc[trainingIdxs]['label'].values

    # Data to test with
    testingText = sentiData.iloc[testingIdxs]['text'].values
    testingLabels = sentiData.iloc[testingIdxs]['label'].values

    sentiPipe.fit(trainingText, trainingLabels)

    # Now, validate the data
    predictions = sentiPipe.predict(testingText)
    sentfusionMatrix  += metrics.confusion_matrix(testingLabels, predictions)
    f1Score = metrics.f1_score(testingLabels, predictions, pos_label='pos')
    sentiScores.append(f1Score)

print("F1 Score: %0.4f" % (sum(sentiScores)/len(sentiScores)))
print("\nConfusion Matrix:")
print(sentfusionMatrix)

print("OK, now on to the SPAM/HAM classifier\n")

class SentiTransform(TransformerMixin):
    def transform(self, X, **transform_params):
        things = []
        for x in X:
            that = sentiPipe.predict([x])
            if that == ['pos']:
                that = [1]
            else:
                that = [0]

            things.append(that)
        return things

    def fit(self, X, y=None, **fit_params):
        return self

start = time.time()
pipe = Pipeline([
    ('features', FeatureUnion([
        ('char_count', CharacterCountTransform()),     # Number of characters in string
        ('$_count', DollarSignCountTransform()),       # Number of `$` character occurances
        ('word_counts', CountVectorizer(stop_words='english')),            # Word token occurance counts
        #('tf_idf_vec', TfidfVectorizer(stop_words='english',min_df=10,max_df = 0.8, sublinear_tf=True, use_idf=True)),
        ('sentiment', SentiTransform()),
        ('numeric_count', NumericTokenTransform()),    # number token occurance counts
        ('alpha_count', AlphaTokenTransform())         # alphabetic token occurance counts
    ])),
    ('classifier', MultinomialNB())                    # The classifier itself
])
end = time.time()
duration = end-start
print("Feature extraction took %.2fsec\n") % (duration)


################
# Train the classifier, using a portion of the data
################
print("=======================")
print("Cross-Validating classifier...")
print("=======================")

# Cross-Validation with 10 folds
kFold = KFold(n=len(dataz['text']), n_folds=10)
f1Scores = []
confusionMatrix = numpy.array([[0,0], [0,0]])
n=0
for trainingIdxs, testingIdxs in kFold:
    # Data to train with
    trainingText = dataz.iloc[trainingIdxs]['text'].values
    trainingLabels = dataz.iloc[trainingIdxs]['label'].values

    # Data to test with
    testingText = dataz.iloc[testingIdxs]['text'].values
    testingLabels = dataz.iloc[testingIdxs]['label'].values

    pipe.fit(trainingText, trainingLabels)

    # Now, validate the data
    predictions = pipe.predict(testingText)
    confusionMatrix  += metrics.confusion_matrix(testingLabels, predictions)
    f1Score = metrics.f1_score(testingLabels, predictions, pos_label=SPAM)
    f1Scores.append(f1Score)

    print("Fold %d Test Report:" % n)
    print("------------")
    print(metrics.classification_report(testingLabels, predictions, target_names=[SPAM,HAM]))
    n = n+1


print("=======================")
print("Aggregate testing results")
print("=======================")

print("F1 Score: %0.4f" % (sum(f1Scores)/len(f1Scores)))
print("\nConfusion Matrix:")
print(confusionMatrix)

# A little bit of post-processing
numSpams = 0
numHams = 0
for i in dataz['label']:
    if i == SPAM:
        numSpams = numSpams + 1
    elif i == HAM:
        numHams = numHams + 1

print("\nTotal SPAM messages:\t\t%d" % numSpams)
print("Total HAM messages:\t\t%d" % numHams)

# SC = Spams Caught = False negative cases / Number of spams
scScore = confusionMatrix[1][1] / float(numSpams)
saScore = confusionMatrix[1][0] / float(numSpams)

# BH = Blocked Hams = False positive cases / Number of hams
bhScore = confusionMatrix[0][1] / float(numHams)

print("\nSpams Caught (True positives)\tSC=%0.4f" % scScore)
print("Spams Allowed (False negatives)\tSA=%0.4f" % saScore)
print("\nBlocked Hams (False positives)\tBH=%0.4f" % bhScore)

hamText = "I'm going to rock this CIS700 Final!"
print("%s -> %s" % (hamText, pipe.predict([hamText])))

