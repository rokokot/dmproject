import os
import pandas as pd
import re
from bs4 import BeautifulSoup

"""
Code to read html files and process them for easier feature extraction.

Args:

Outs:


"""

def read_files(file_path):    # quick file reader to string
  try:
    with open(file_path, 'r') as file:
      content = file.read()
    return content
  
  except Exception as e:
    print(f'error reading {file_path}')
    return ""
  

def clean_html(html):   # very similar to exercise code

  soup = BeautifulSoup(html,'html.parser')

  for script in soup(['script', 'style']):
    script.decompose()
  
  text = soup.get_text()

  lines = (line.strip() for line in text.splitlines())
  chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
  text = ' '.join(chunk for chunk in chunks if chunk)     # beautiful loop
  return text

def extract_title(html):    # important feature to look at
  "get first heading from source"
  soup = BeautifulSoup(html, 'html.parser')

  h1 = soup.find('h1')
  
  if h1:
    return h1.text.strip()

  title = soup.find('title')
  if title:
    return title.text.strip()
  

  return ""

def extract_headings(html):   # rational behind this

def extract_links(html):

def extract_metadata(html):


def extract_objects(html):

def extract_keywords(html):


def process_html(html, file=""):

  clean_text = clean_html(html)
  title = extract_title(html)
  metadata = extract_metadata(html)
  headings = extract_headings(html)
  links = extract_links(html)
  objects = extract_objects(html)
  keywords = extract_keywords(html)

  processed_html = {
    'filename': file,
    'sourcecode': html,
    'clean_text': clean_text,
    'text_length': len(clean_text),
    'html_length': len(html),
    **metadata,
    **keywords,
    'title': title,
    'headings': {k: len(v) for k, v in headings.items()},
    'num_total_headings': sum(len(v) for v in headings.values()),
    'num_links': len(links),
    'num_out_links': sum(1 for link in links if link['href'].startswith('http')),
    'num_emails': sum(1 for link in links if link['href'].startswith('mailto:')),
    'page_type_keywords': {k: v['count'] for k, v in keywords.items()}

  }

  return processed_html



def load_datasets(dir):   # function to read data from html files, and annotate our train data based on subdir name
  
  data = []
  labels = []

  if not os.path.exists(dir):
    print(f'error, {dir} doesnt exist')
    return pd.DataFrame(), []
  
  subdirs = ['course', 'faculty', 'student']

  has_subdirs = all(os.path.exists(os.path.join(dir, subdir)) for subdir in subdirs)

  if has_subdirs:
    print(f"Loading training data from {dir}")
    for label in subdirs:
      subdir_path = os.path.join(dir, label)
      files = os.listdir(subdir_path)
      print(f"  {label}: {len(files)} files")
      
      for filename in files:
          if filename.endswith('.html'):
            file_path = os.path.join(subdir_path, filename)
            content = read_files(file_path)
            
            processed = process_html(content, filename)
            processed['label'] = label
            
            data.append(processed)
            labels.append(label)
  else:
    print('loading test data')
    files = os.listdir(dir)
    print(f'found {len(files)} at {dir}')

    for filename in files:
      if filename.endswith('.html'):
        file_path = os.path.join(dir, filename)
        content = read_files(file_path)

        processed = process_html(content, filename)
        processed['label'] = 'unk'      # unk labels to test, combine features to a dataset
        
        data.append(processed)
        labels.append('unk')
  
  return pd.DataFrame(data), labels


def extract_features(data):     # creates a dict for easy value retrieval, stores results of collection functions above

  features = []


  return pd.DataFrame(features)
  




