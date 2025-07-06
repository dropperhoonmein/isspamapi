from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessedText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def fit(self,X,Y=None):
        return self
    def transform(self,X):
        processedText = []
        for text in X:
            words = text.lower().split(" ")
            words = [self.stemmer.stem(word) for word in words if word not in ENGLISH_STOP_WORDS]
            processedText.append(' '.join(words))
        return processedText