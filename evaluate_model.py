import pandas as pd
import pickle
import numpy as np
import time
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
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
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    models = {
        'dummy': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(max_iter=2000, solver='liblinear'),
        'xgboost': xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, verbosity=1),
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=True)}
    
    trained_models = {}

    print(f"--------------------")
    print("Training final models on full dataset...")
    print(f"--------------------")
    
    for name, model in models.items():
        print(f"running  {name}...", end='', flush=True)
        start_time = time.time()
        
        model.fit(features, encoded_labels)
        
        duration = time.time() - start_time
        print(f" completed in {duration} seconds")
        
        trained_models[name] = (model, le)
    
    return trained_models


def evaluate_models(models, test_features, test_labels=None):
    results = {}
    
    for name, (model, le) in models.items():
        encoded_predictions = model.predict(test_features)

        predictions = le.inverse_transform(encoded_predictions)
        results[name] = predictions
        
        if test_labels is not None:
            accuracy = accuracy_score(test_labels, predictions)
            print(f"\n{name} accuracy: {accuracy:.4f}")
            print(classification_report(test_labels, predictions))
    
    return results


def cv(features, labels):
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    models = {'dummy': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': LogisticRegression(max_iter=2000, solver='liblinear'),
        'xgboost': xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, verbosity=1),
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=True)}
    
    cv_results = {}
    start = time.time()
    
    for name, model in models.items():
        print('---------------------------')
        print(f'started {name} evaluation')
        print('---------------------------')

        start_time = time.time()
        
       
        scores = cross_val_score(model, features, encoded_labels, cv=10)
      
        end_time = time.time()
        duration = end_time - start_time
        
        cv_results[name] = scores
        
        print(f"\n{name}:")     # log experiment finish
        print(f"  Time taken: {duration:.1f} seconds")
        print(f"  Individual fold scores:")
        
        for i, acc in enumerate(scores, 1):
            print(f"    Fold {i:2d}: {acc:.4f}")
      
        print(f"Mean accuracy: {scores.mean():.4f}")
    
    total_duration = time.time() - start
    print(f"cross-validation duration: {total_duration:.1f} seconds")

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


