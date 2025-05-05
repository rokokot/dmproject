import pandas as pd
import pickle
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from scipy.stats import ttest_rel


"""
Main training script for our classifier models..

Args:
- features, x_train content features, and y_train labels

"""

def train_models(features, labels):
    models = {
       
        'dummy': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(max_iter=200),
        'xgboost': xgb.XGBClassifier(objective='multi:softprob', num_class=3),
        'mlp': MLPClassifier(hidden_laayer_size=(100,), max_iter=1000)}
    
    trained_models = {}
    for name, model in models.items():
        model.fit(features, labels)
        trained_models[name] = model
    
    return trained_models


def evaluate_models(models, test_features, test_labels=None):
    results = {}
    
    for name, model in models.items():
        predictions = model.predict(test_features)
        results[name] = predictions
        
        if test_labels is not None:
            accuracy = accuracy_score(test_labels, predictions)
            print(f"\n{name} accuracy: {accuracy:.4f}")
            print(classification_report(test_labels, predictions))
    
    return results


def cv(features, labels):
    models = {
        'dummy': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(max_iter=200),
        'xgboost': xgb.XGBClassifier(objective='multi:softprob', num_class=3),
        'mlp': MLPClassifier(hidden_laayer_size=(100,), max_iter=1000)}
    
    cv_results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, features, labels, cv=10)
        cv_results[name] = scores
        
        print(f"\n{name}:")
        for acc in scores:
            print(f"  {acc:.4f}")
        print(f"Mean accuracy: {scores.mean():.4f}")
    
    return cv_results


def t_test(cv_results):     # scipy reference implementation
    
    models = list(cv_results.keys())
    results = {}
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            scores1 = cv_results[model1]
            scores2 = cv_results[model2]
            
            t_stat, p_value = ttest_rel(scores1, scores2)
            
            results[f"{model1} vs {model2}"] = {
                'mean_diff': np.mean(scores1) - np.mean(scores2),
                't_stat': t_stat,
                'p_value': p_value}
    
    return results


