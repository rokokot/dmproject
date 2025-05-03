import os
import re
from bs4 import BeautifulSoup

"""
Code to read html files and process them for easier feature extraction.

Args:

Outs:


"""

def read_files(file_path):

  try:
    with open(file_path, 'r') as file:
      content = file.read()
    return content
  
  except Exception as e:
    print(f'error reading {file_path}')
    return ""
  

def clean_html(html):
  return text

def extract_heading(html):
  return string 

def load_datasets(dir):
  return data, labels

