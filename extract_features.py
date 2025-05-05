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
        ('h1_count', Pipeline([
            ('selector', ItemSelector(key='h1_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('h2_count', Pipeline([
            ('selector', ItemSelector(key='h2_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('h3_count', Pipeline([
            ('selector', ItemSelector(key='h3_count', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('total_headings', Pipeline([
            ('selector', ItemSelector(key='total_headings', to_dict=True)),
            ('sparse', DictVectorizer(sparse=True)),
        ])),
        ('link_count', Pipeline([
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
        ('hr_count', Pipeline([
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
  
    
feature_df = extract_features(data)

feature_matrix = features.fit_transform(feature_df)

return feature_matrix, features, None, None

