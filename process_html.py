import os
import pandas as pd
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

  soup = BeautifulSoup(html,'html.parser')

  for script in soup(['script', 'style']):
    script.decompose()
  return text

def extract_heading(html):
  return string 

def load_datasets(dir):
  
  data = []
  labels = []

  if not os.path.exists(dir):
    print(f'error, {dir} doesnt exist')
    return pd.DataFrame(), []
  
  for file in os.listdir(dir):
    if file.endswith('.html'):
      file_path = os.path.join(dir, file)
      content = read_files(file_path)

      label = file.split('_')[0]

      data.append({
        'filename': file,
        'html': content,
        'text': clean_html(content)
      })
      labels.append(label)

  return pd.DataFrame(data), labels



