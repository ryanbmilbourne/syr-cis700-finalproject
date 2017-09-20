#!/usr/bin/python

#Helpful functions to process our data

import processData
import numpy as np
import sys
import re
# FU Python
sys.path.append('/usr/local/lib/python2.7/site-packages')

# External imports
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
import numpy
from numpy import vectorize
import pandas as panduh

# Internal imports
from processData import processData

# Helpful constants
SPAM = 'spam'
HAM = 'ham'

DATAFILE_PATH = './dataSet/SMSSpamCollection.txt' # Relative path to file

#####################
# getCharacterCount
#####################
# Function that processes a panda Dataframe
# and appends character count feature
#
# @param X ndarray
#
# @return feature vector with character count, minus whitespace
##
class CharacterCountTransform(TransformerMixin):
    def transform(self, X, **transform_params):
        return [[len(re.findall('[a-z]?[A-Z]?[0-9]?', x))] for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

class DollarSignCountTransform(TransformerMixin):
    def transform(self, X, **transform_params):
        return [[x.count('$')] for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

class NumericTokenTransform(TransformerMixin):
    def transform(self, X, **transform_params):
        return [[len(re.findall('[0-9]+', x))] for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

class AlphaTokenTransform(TransformerMixin):
    def transform(self, X, **transform_params):
        return [[len(re.findall('[A-Za-z]+', x))] for x in X]

    def fit(self, X, y=None, **fit_params):
        return self

#####################
# FOR TESTING
#####################
if __name__ == "__main__":
    ################
    # Fetch the data
    ################
    print("Reading in SMS data...")
    #dataz = processData(DATAFILE_PATH)
    #getCharacterCount(dataz)

