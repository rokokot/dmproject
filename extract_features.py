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


"""
class ItemSelector(BaseEstimator, TransformerMixin):
  def __init__(self, key, to_dict=False):
    self.key = key 
    self.to_dict = to_dict

  def fit(self, x, y=None):
    return self
  
  def transform(self, data_dict):
    if self.to_dict:
      return [{self.key: data_dict.loc[i, self.key]} for i in data_dict.index]
    else:
      return data_dict.loc[:, self.key]

def create_combined_features(data, labels):
    
    feature_df = extract_features(data)
    raw_data = data
    # define a  feature selection pipeline similar to the exercises

    features = FeatureUnion(transformer_list=[
        ('text', Pipeline([
            ('selector', ItemSelector(key='clean_text')),
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ])),
        ('title', Pipeline([
            ('selector', ItemSelector(key='title')),
            ('counts', CountVectorizer(stop_words='english', max_features=200)),
        ])),
        ('clean_', Pipeline([
          ('selector', ItemSelector(key='text_length', to_dict=True)),
          ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('full', Pipeline([
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
        ('he', Pipeline([
            ('selector', ItemSelector(key='total_headings', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('l', Pipeline([
            ('selector', ItemSelector(key='link_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('ex', Pipeline([
            ('selector', ItemSelector(key='external_links', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('e', Pipeline([
            ('selector', ItemSelector(key='email_links', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('ph', Pipeline([
            ('selector', ItemSelector(key='phone_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('n', Pipeline([
            ('selector', ItemSelector(key='has_course_number', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('p', Pipeline([
            ('selector', ItemSelector(key='paragraph_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('h', Pipeline([
            ('selector', ItemSelector(key='hr_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('i', Pipeline([
            ('selector', ItemSelector(key='image_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('sk', Pipeline([
            ('selector', ItemSelector(key='student_keywords', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('fk', Pipeline([
            ('selector', ItemSelector(key='faculty_keywords', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('ck', Pipeline([
            ('selector', ItemSelector(key='course_keywords', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),])
  
  
    combined_data = raw_data.copy()
    for col in feature_df.columns:
        combined_data[col]=feature_df[col]

    feature_matrix = features.fit_transform(combined_data)


    feature_names = []
    
    try:
        for name, transformer in features.transformer_list:

            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out()
                    feature_names.extend([f"{name}__{n}" for n in names])
                    continue
                except:
                    pass
                
            if hasattr(transformer, 'steps'):
                last_step = transformer.steps[-1][1]
                if hasattr(last_step, 'get_feature_names_out'):
                    try:
                        names = last_step.get_feature_names_out()
                        feature_names.extend([f"{name}__{n}" for n in names])
                        continue
                    except:
                        pass
            
            n_features = transformer.transform(combined_data).shape[1]
            feature_names.extend([f"{name}_{i}" for i in range(n_features)])
    except:
        feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]

    if hasattr(feature_matrix, 'toarray'):
        dense_matrix = feature_matrix.toarray()
    else:
        dense_matrix = feature_matrix

    feature_df_export = pd.DataFrame(dense_matrix,  columns=feature_names, index=data['filename'])

    feature_df_export.to_csv('featurematrix.csv')
    print(f"Feature matrix exported to featurematrix.csv ({feature_matrix.shape[0]} rows, {feature_matrix.shape[1]} columns)")

    return feature_matrix, features, None, None

