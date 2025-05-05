import os
import argparse
import pandas as pd
import pickle
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
  parser.add_argument('--cache_dir', default='./cache', help='enter path to cache dir')
  parser.add_argument('--use_cache', action='store_true', help='use cached data if it is available, else save data to cache after loading')
    
  args = parser.parse_args()

  os.makedirs('results', exist_ok=True)
  os.makedirs(args.cache_dir, exist_ok=True)


  print("Running html webpage classification project demo: ")
  
  print("loading training data...")

  train_cache_file = os.path.join(args.cache_dir, 'processed_train.pkl')
  test_cache_file = os.path.join(args.cache_dir, 'processed_test.pkl')

  print("Loading training data...")
  
  if args.use_cache:

    cached_data = load_cached_data(train_cache_file)
    if cached_data is not None:

      print(f"Loading training data from cache: {train_cache_file}")

      train_data, train_labels = cached_data
    else:

      print(f"Training cache not found. Loading from source and saving to cache...")

      train_data, train_labels = load_datasets(args.train_dir)

      cache_processed_data((train_data, train_labels), train_cache_file)
  else:
      
      print("Loading training data from source (cache disabled)")
      train_data, train_labels = load_datasets(args.train_dir)

      cache_processed_data((train_data, train_labels), train_cache_file)
      print(f"Saved training data to cache: {train_cache_file}")
  
  print("\nLoading test data...")
  if args.use_cache:

    cached_data = load_cached_data(test_cache_file)
    if cached_data is not None:

      print(f"Loading test data from cache: {test_cache_file}")
      test_data, test_labels = cached_data
    else:

      print(f"Test cache not found. Loading from source and saving to cache...")

      test_data, test_labels = load_datasets(args.test_dir)
      cache_processed_data((test_data, test_labels), test_cache_file)
  else:
      print("Loading test data from source (cache disabled)")
      test_data, test_labels = load_datasets(args.test_dir)
      
      cache_processed_data((test_data, test_labels), test_cache_file)
      print(f"Saved test data to cache: {test_cache_file}")
    

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
  
  
  # python main.py will always load from source and save to cache

  # python main.py --cache_dir ./custm/cache --use_cache will use a custom cache folder

  #python main.py ---train_dir ./data/train --test_dir ./data/test --cache_dir /cache --use_cache

