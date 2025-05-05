import os
import argparse
import pandas as pd
from process_html import load_datasets, load_cached_data, cache_processed_data
from extract_features import create_combined_features
from evaluate_model import *

"""
Main experiment runner for the pipeline, 

Args:
- train dir
- test dir
- optional cache dir for quicker processing
"""

def main(): # run function with arguments

  parser = argparse.ArgumentParser(description='Intro to Data Mining Project, Webpage (html) classification')

  parser.add_argument('--train_dir', default='./data/train', help='enter path to train data')
  parser.add_argument('--test_dir', default='./data/test', help='enter path to test data')
  args = parser.parse_args()

  os.makedirs('results', exist_ok=True)

  print("Running html webpage classification pipeline: ")
  
  print("Loading training data...")

  cache_file = f'./cache/processed_train.pkl' if args.train_dir == './data/train' else f'data/processed_test.pkl'

  cached_data = load_cached_data(cache_file)

  if cached_data is not None:
      print(f"Loading from cache: {cache_file}")
      train_data, train_labels = cached_data

  else:
      train_data, train_labels = load_datasets(args.train_dir)
      cache_processed_data((train_data, train_labels), cache_file)

  
  print("\nLoading test data...")
  test_data, _ = load_datasets(args.test_dir)
  
  print("\nExtracting features...")
  X_train, feature_pipeline, _, _ = create_combined_features(train_data, train_labels)
  X_test, _, _, _ = create_combined_features(test_data, None)
  
  print("\nPerforming 10-fold cross-validation...")
  cv_results = cv(X_train, train_labels)
  
  print("\nPerforming t-tests...")
  t_test_results = t_test(cv_results)
  
  print("\nresults:")
  for comparison, stats in t_test_results.items():
      print(f"\n{comparison}:")
      print(f"  Mean difference: {stats['mean_diff']:.4f}")
      print(f"  T-statistic: {stats['t_stat']:.4f}")
      print(f"  P-value: {stats['p_value']:.4f}")
  
  best_model_name = max(cv_results.items(), key=lambda x: x[1].mean())[0]
  print(f"\nbest model: {best_model_name}")
  
  trained_models = train_models(X_train, train_labels)
  
  test_predictions = evaluate_models(trained_models, X_test)
  
  results_df = pd.DataFrame({'filename': test_data['filename'].values})
  for model_name, predictions in test_predictions.items():
      results_df[f'{model_name}_prediction'] = predictions
  
  results_df.to_csv('results/test_predictions.csv', index=False)
  
  with open(f'results/best_model_{best_model_name}.pkl', 'wb') as f:
      pickle.dump(trained_models[best_model_name], f)
  
  print("\nResults saved to 'results' directory")


if __name__ == '__main__':
  main()
  