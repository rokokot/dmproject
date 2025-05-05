#!/usr/bin/env python
from process_html import load_datasets, extract_features
import pandas as pd

train_data, train_labels = load_datasets('./data/train')

print("running html processing tests:")

print(f"loaded {len(train_data)} training examples")

if len(train_data) > 0:
  
  example = train_data.iloc[0]    # take the first example
  print('first data example contains the following features:')
  print(f'example: {example["filename"]}')

  print(f'title: {example["title"]}')
  print(f'text len: {example["text_length"]}')
  print(f'headings: {example["num_total_headings"]}')
  print()

feature_df = extract_features(train_data)
print(f'features df dimensions: {feature_df.shape}')
print(f'columns: {list(feature_df.columns)}')

test_data, _ = load_datasets('./data/test')   # placeholder for test labels
print(f'loaded {len(test_data)} examples')

