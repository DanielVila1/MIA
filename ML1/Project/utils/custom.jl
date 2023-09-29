
py"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import preprocessing
import numpy as np

class CustomClassifier(BaseEstimator, ClassifierMixin):

    binary_classifier=None
    multi_classifier=None

    def __init__(self, binary_classifier=None, multi_classifier=None):
        self.binary_classifier = binary_classifier
        self.multi_classifier = multi_classifier

    def fit(self, X, y=None):
        ii = np.where(y > 0)
        bin_y = np.array(y, copy=True)
        bin_y[ii] = 1.0
        multi_X = X[ii]
        multi_y = y[ii] 
        self.binary_classifier.fit(X, bin_y)
        self.multi_classifier.fit(multi_X, multi_y)
    
    def predict(self, X, y=None):
        X32 = X.astype('float32')
        bin_pred = self.binary_classifier.predict(X32)
        multi_pred = self.multi_classifier.predict(X32)
        final_pred = np.multiply(bin_pred, multi_pred)
        return final_pred

    def predict_proba(self, X, y=None):
        bin_pred = self.binary_classifier.predict(X)
        bin_pred_complement = 1 - bin_pred
        bin_pred_proba = self.binary_classifier.predict_proba(X)
        multi_pred_proba = self.multi_classifier.predict_proba(X)
        final_bin_pred_proba = np.multiply(bin_pred_complement, bin_pred_proba)
        final_multi_pred_proba = np.multiply(np.multiply(bin_pred, bin_pred_proba), multi_pred_proba)
        final_pred = final_bin_pred_proba + final_multi_pred_proba
        return final_pred

"""
