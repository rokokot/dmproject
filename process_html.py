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
  
  text = soup.get_text()

  lines = (line.strip() for line in text.splitlines())
  chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
  text = ' '.join(chunk for chunk in chunks if chunk)     # beautiful loop
  return text

def extract_title(html):
  "get first heading from source"
  soup = BeautifulSoup(html, 'html.parser')

  h1 = soup.find('h1')
  
  if h1:
    return h1.text.strip()

  title = soup.find('title')
  if title:
    return title.text.strip()
  

  return ""

def extract_headings(html):

def extract_links(html):

def extract_metadata(html):


def extract_objects(html):

def extract_keywords(html):


def process_html_advanced(html, file=""):

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



