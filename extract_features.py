import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from bs4 import BeautifulSoup
from scipy.sparse import hstack

from process_html import extract_features

"""
Feature extraction module which runs the preprocessing steps on our html feature data , and returns several different types of features for training classifiers.

Args:
  - file paths

Outs:
    - sk FeatureUnion
    - feature matrix
"""

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key, to_dict=False):
        self.key = key 
        self.to_dict = to_dict

    def fit(self, x, y=None):
        return self
  
    def transform(self, data_dict):
        if self.to_dict:
            result = []
            for i in data_dict.index:
                value = data_dict.loc[i, self.key]
                if pd.isna(value):
                    value = '' if isinstance(value, str) else 0
                result.append({self.key: value})
            return result
        else:
            # Get the column and replace NaN values
            column = data_dict.loc[:, self.key]
            if column.dtype == 'object' or column.dtype == 'string':
                return column.fillna('').astype(str).values
            else:
                return column.fillna(0).values

def create_features():
    return FeatureUnion(transformer_list=[
        ('text', Pipeline([
            ('selector', ItemSelector(key='clean_text')),
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ])),
        ('title', Pipeline([
            ('selector', ItemSelector(key='title')),
            ('counts', CountVectorizer(stop_words='english', max_features=200)),
        ])),
        ('text_length', Pipeline([
            ('selector', ItemSelector(key='text_length', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('html_length', Pipeline([
            ('selector', ItemSelector(key='html_length', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('h1', Pipeline([
            ('selector', ItemSelector(key='h1_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('h2', Pipeline([
            ('selector', ItemSelector(key='h2_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('h3', Pipeline([
            ('selector', ItemSelector(key='h3_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('heading_counts', Pipeline([
            ('selector', ItemSelector(key='total_headings', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('link_counts', Pipeline([
            ('selector', ItemSelector(key='link_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('external_links', Pipeline([
            ('selector', ItemSelector(key='external_links', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('email_links', Pipeline([
            ('selector', ItemSelector(key='email_links', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('phone_count', Pipeline([
            ('selector', ItemSelector(key='phone_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('has_course_number', Pipeline([
            ('selector', ItemSelector(key='has_course_number', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('paragraph_count', Pipeline([
            ('selector', ItemSelector(key='paragraph_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('rule_count', Pipeline([
            ('selector', ItemSelector(key='hr_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('image_count', Pipeline([
            ('selector', ItemSelector(key='image_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('student_keywords', Pipeline([
            ('selector', ItemSelector(key='student_keywords', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('faculty_keywords', Pipeline([
            ('selector', ItemSelector(key='faculty_keywords', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('course_keywords', Pipeline([
            ('selector', ItemSelector(key='course_keywords', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
    ])
  
def extract_features_from_data(data, feature_pipeline=None):
    feature_df = extract_features(data)
    
    combined_data = data.copy()
    for col in feature_df.columns:
        combined_data.loc[:, col] = feature_df[col]
    
    if feature_pipeline is None:
        feature_pipeline = create_features()
        features = feature_pipeline.fit_transform(combined_data)
    else:
        features = feature_pipeline.transform(combined_data)
    
    return features, feature_pipeline

def export_feature_matrix(features, filenames, labels=None, split_type=None):
    if hasattr(features, 'toarray'):
        dense_matrix = features.toarray()
    else:
        dense_matrix = features
    
    n_features = dense_matrix.shape[1]
    feature_names = [f'feature_{i}' for i in range(n_features)]

    feature_df = pd.DataFrame(dense_matrix, columns=feature_names, index=filenames)
    
    if labels:
        feature_df['label'] = labels
    
    if split_type:
        feature_df['split'] = split_type
    
    feature_df.to_csv('featurematrix.csv')
    print(f"feature matrix stats => ({features.shape[0]})")