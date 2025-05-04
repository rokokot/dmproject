#!/bin/usr/
from process_html import load_datasets, extract_html_features
import pandas as pd

train_data, train_labels = load_datasets('./data/train')


example = train_data.iloc[0]


feature_df = extract_html_features(train_data)

test_data = load_datasets('./data/test')