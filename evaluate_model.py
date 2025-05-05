import pandas as pd
import pickle
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

"""
Main training script for our classifier models..

Args:
- features, x_train content features, and y_train labels


"""

def train_models(features):
  return model 


def evaluate_models(model, test):
  return results


def cv(features, labels):
  models = {
        'dummy': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(max_iter=200),
        'xgboost': xgb.XGBClassifier(objective='multi:softprob'),
        'perceptron': Perceptron(max_iter=k)}
    
  cv_results = {}

  for name, model in models.items():
      
      scores = cross_val_score(model, features, labels, cv=10)
      cv_results[name] = scores

      print(f"\n{name}:")

      for acc in scores:
          print(acc)
      print(f"Mean accuracy: {scores.mean():.4f}")
  
  return cv_results



def t_test(results):    # scipy ttests
  return t_results

