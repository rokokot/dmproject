import pandas as pd
import pickle
import numpy as np
import time
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from scipy.stats import ttest_rel

from extract_features import extract_features_from_data

"""
Main training script for our classifier models..

Args:
- features, x_train content features, and y_train labels

Outs:
"""

RANDOM_SEED = 69
np.random.seed(RANDOM_SEED)


def train_models(features, labels):
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    models = {
        'dummy': DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED),
        'logistic_regression': LogisticRegression(max_iter=2000, solver='liblinear', random_state=RANDOM_SEED),
        'xgboost': xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, verbosity=1, random_state=RANDOM_SEED),
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=True, random_state=RANDOM_SEED)
    }
    
    trained_models = {}

    print(f"--------------------")
    print("Training each model on full dataset...")
    print(f"--------------------")
    
    for name, model in models.items():
        print(f"running  {name}...", end='', flush=True)
        start_time = time.time()
        
        model.fit(features, encoded_labels)
        
        duration = time.time() - start_time
        print(f" completed in {duration:.2f} seconds")
        
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
            precision = precision_score(test_labels, predictions, average='weighted')
            recall = recall_score(test_labels, predictions, average='weighted')
            
            print(f"{name} performance metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(classification_report(test_labels, predictions))
    
    return results


def cross_validate(data, labels, n_folds=10):
    # First encode all labels - but we'll use a common encoder to avoid issues
    le = LabelEncoder()
    le.fit(labels)  # Make sure encoder knows all possible classes
    
    # Define models with fixed random seeds
    models = {
        'dummy': DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED),
        'logistic_regression': LogisticRegression(max_iter=2000, solver='liblinear', random_state=RANDOM_SEED),
        # Set verbosity to 0 for XGBoost to reduce output
        'xgboost': xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, verbosity=0, random_state=RANDOM_SEED),
        'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=False, random_state=RANDOM_SEED)
    }
    
    # Create encoded labels array for stratified splits
    encoded_labels = le.transform(labels)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    cv_results = {}
    metrics_results = {}
    start = time.time()
    
    for name, model in models.items():
        print('---------------------------')
        print(f'started {name} evaluation')
        print('---------------------------')

        start_time = time.time()
        
        accuracies = np.zeros(n_folds)
        precisions = np.zeros(n_folds)
        recalls = np.zeros(n_folds)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, encoded_labels)):
            fold_num = fold_idx + 1
            print(f"  Processing fold {fold_num}/{n_folds}...")
            
            # Get data for this fold
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Get the encoded labels directly from our encoded array
            train_labels_enc = encoded_labels[train_idx]
            test_labels_enc = encoded_labels[test_idx]
            
            # Get original string labels for test set (needed for evaluation)
            test_labels_list = [labels[i] for i in test_idx]
            test_labels_array = np.array(test_labels_list)
            
            print(f"    getting features for fold {fold_num}...")
            X_train, feature_pipeline = extract_features_from_data(train_data)
            X_test, _ = extract_features_from_data(test_data, feature_pipeline)
            
            print(f"    Training {name} for fold {fold_num}...")
            
            # Train model using encoded labels to avoid XGBoost issues
            model.fit(X_train, train_labels_enc)
            
            # Make predictions
            y_pred_enc = model.predict(X_test)
            
            # Convert predictions back to original labels for evaluation
            y_pred = le.inverse_transform(y_pred_enc)
            
            # Calculate metrics using original labels
            accuracies[fold_idx] = accuracy_score(test_labels_array, y_pred)
            precisions[fold_idx] = precision_score(test_labels_array, y_pred, average='weighted')
            recalls[fold_idx] = recall_score(test_labels_array, y_pred, average='weighted')
            
            print(f"    Fold {fold_num} results: Acc={accuracies[fold_idx]:.4f}, "
                  f"Prec={precisions[fold_idx]:.4f}, Rec={recalls[fold_idx]:.4f}")
        
        cv_results[name] = accuracies
        metrics_results[name] = {
            'accuracy': accuracies,
            'precision': precisions,
            'recall': recalls}
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{name}:")
        print(f"  Time taken: {duration:.1f} seconds")
        print(f"  Individual fold scores:")
        
        for i in range(n_folds):
            print(f"    Fold {i+1:2d}: Acc={accuracies[i]:.4f}, "
                  f"Prec={precisions[i]:.4f}, Rec={recalls[i]:.4f}")
        
        print(f"  Mean metrics across folds:")
        print(f"    Accuracy:  {np.mean(accuracies):.2f}")
        print(f"    Precision: {np.mean(precisions):.2f}")
        print(f"    Recall:    {np.mean(recalls):.2f}")
    
    total_duration = time.time() - start
    print(f"cross-validation duration: {total_duration:.1f} seconds")

    return cv_results, metrics_results


def t_test(cv_results):
    models = list(cv_results.keys())
    results = {}
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            scores1 = cv_results[model1]
            scores2 = cv_results[model2]
            
            result = ttest_rel(scores1, scores2)
            
            results[f"{model1} vs {model2}"] = {
                'mean_delta': np.mean(scores1) - np.mean(scores2),
                't_stat': result.statistic
            }
      
    return results