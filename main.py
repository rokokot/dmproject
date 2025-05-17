import os
import argparse
import pandas as pd
import pickle
import numpy as np
from process_html import load_datasets, load_cached_data, cache_processed_data
from extract_features import extract_features_from_data
from evaluate_model import (
    train_models, 
    evaluate_models, 
    t_test,
    cross_validate,
    RANDOM_SEED
)

"""
Main experiment runner for the pipeline, 

Args:
- train dir
- test dir
- optional cache dir for quicker processing
"""

np.random.seed(RANDOM_SEED)

def main():
    parser = argparse.ArgumentParser(description='webpage (html) classification')
    parser.add_argument('--train_dir', default='./data/train', help='path to training data')
    parser.add_argument('--test_dir', default='./data/test', help='path to test data')
    parser.add_argument('--cache_dir', default='./cache', help='path to cache directory')
    parser.add_argument('--use_cache', action='store_true', help='use cached data if available')
    parser.add_argument('--skip_cv', action='store_true', help='skip cross-validation')
    parser.add_argument('--n_folds', type=int, default=10, help='number of cross-validation folds')
    
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("\n===== Webpage Classification Demo =====")
    print(f"Random seed: {RANDOM_SEED}")
    
    print("Loading train data...")
    train_cache_file = os.path.join(args.cache_dir, 'processed_train.pkl')
    
    if args.use_cache and os.path.exists(train_cache_file):
        print(f"  Loading from cache: {train_cache_file}")
        train_data, train_labels = load_cached_data(train_cache_file)
    else:
        print(f"  Loading from source: {args.train_dir}")
        train_data, train_labels = load_datasets(args.train_dir)
        cache_processed_data((train_data, train_labels), train_cache_file)
    
    print(f"  Loaded {len(train_data)} training examples")
    
    print("\nLoading test data...")
    test_cache_file = os.path.join(args.cache_dir, 'processed_test.pkl')
    
    if args.use_cache and os.path.exists(test_cache_file):
        print(f"  Loading from cache: {test_cache_file}")
        test_data, test_labels = load_cached_data(test_cache_file)
    else:
        print(f"  Loading from source: {args.test_dir}")
        test_data, test_labels = load_datasets(args.test_dir)
        cache_processed_data((test_data, test_labels), test_cache_file)
    
    print(f"  Loaded {len(test_data)} test examples")
    

    if not args.skip_cv:
        print("\n===== Cross-Validation =====")
        print(f"starting {args.n_folds}-fold cross-validation")
        
        cv_results, metrics_results = cross_validate(train_data, train_labels, args.n_folds)
        
        with open('results/cv_results.pkl', 'wb') as f:
            pickle.dump(metrics_results, f)
        
      
        summary = []   # for visualization and reporting we keep some more statis per fold
        for model_name, metrics in metrics_results.items():
            for fold in range(len(metrics['accuracy'])):
                summary.append({
                    'Model': model_name,
                    'Fold': fold + 1,
                    'Accuracy': metrics['accuracy'][fold],
                    'Precision': metrics['precision'][fold],
                    'Recall': metrics['recall'][fold]
                })
        
        pd.DataFrame(summary).to_csv('results/summary.csv', index=False)
        
        print("\n===== T Tests =====")
        t_test_results = t_test(cv_results)
        
        t_test_df = []
        for comparison, stats in t_test_results.items():
            print(f"\n{comparison}:")
            print(f"  Mean delta: {stats['mean_delta']:.3f}")
            print(f"  t-statistic: {stats['t_stat']:.3f}")
         
            t_test_df.append({
                'Comparison': comparison,
                'Mean_Delta': stats['mean_delta'],
                't_statistic': stats['t_stat']
            })
        
        pd.DataFrame(t_test_df).to_csv('results/ttest_results.csv', index=False)
        print("\t test results saved to results/ttest_results.csv")
    else:
        print("\nSkipping cross-validation (--skip_cv flag used)")
    
    print("\n===== Training Final Models =====")
    print("Extracting features from full training dataset...")
    X_train, feature_pipeline = extract_features_from_data(train_data)
    print(f"train feature matrix shape: {X_train.shape}")
    
    trained_models = train_models(X_train, train_labels)
    
    
    with open('results/trained_models.pkl', 'wb') as f:         # save model checkpoints and parameters
        pickle.dump(trained_models, f)
    print("Trained models saved to results/trained_models.pkl")
    
# uncomment below block to run predictions on test data
""" 
    print("\n===== Evaluating Test Data =====")
    print("Extracting features from test dataset...")
    X_test, _ = extract_features_from_data(test_data, feature_pipeline)
    print(f"test feature matrix shape: {X_test.shape}")
    
    if hasattr(X_train, 'toarray'):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train
        
    if hasattr(X_test, 'toarray'):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test
    
    all_features = np.vstack([X_train_dense, X_test_dense])
    all_filenames = np.concatenate([train_data['filename'].values, test_data['filename'].values])
    
    n_train = len(train_data)
    n_test = len(test_data)
    split_info = ['train'] * n_train + ['test'] * n_test
    
    has_test_labels = test_labels[0] != 'unk'
    if has_test_labels:
        all_labels = np.concatenate([train_labels, test_labels])
    else:
        all_labels = np.concatenate([train_labels, ['unknown'] * n_test])
    
    feature_names = [f'feature_{i}' for i in range(all_features.shape[1])]
    df = pd.DataFrame(all_features, columns=feature_names)
    df['filename'] = all_filenames
    df['split'] = split_info
    df['label'] = all_labels
    
    df.to_csv('featurematrix.csv', index=False)
    print(f"Combined feature matrix saved to featurematrix.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
    


    print("===== Run prediction on test data =====")
    has_true_labels = test_labels[0] != 'unk'
    true_labels = test_labels if has_true_labels else None
    
    predictions = evaluate_models(trained_models, X_test, true_labels)
    
    results_df = pd.DataFrame({'filename': test_data['filename'].values})
    for model_name, model_predictions in predictions.items():
        results_df[f'{model_name}_prediction'] = model_predictions
    
    # save predictions
    results_df.to_csv('results/test_predictions.csv', index=False)
    print(" predictions saved to results/test_predictions.csv")
    
"""

if __name__ == '__main__':
    main()