import os
import pandas as pd
import re
import pickle
from tqdm import tqdm
from bs4 import BeautifulSoup

"""
Code to read html files and process them for easier feature extraction.

Args:

Outs:


"""

def read_files(file_path):    # quick file reader to string
  try:
    with open(file_path, 'r', encoding='iso-8859-1') as file:
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


def cache_processed_data(data,cache_file):    #reading files and parsing them into cache, to avoid long runtime
   
   with open(cache_file, 'wb') as f:
      pickle.dump(data, f)

def load_cached_data(cache_file):
  if os.path.exists(cache_file):
      with open(cache_file, 'rb') as f:
         return pickle.load(f)
  return None



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
  " count all headings "
  soup = BeautifulSoup(html, 'html.parser')
    
  headings = {'h1': [], 'h2': [], 'h3': [], 'h4': [], 'h5': [], 'h6': []}
    
  for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
    for heading in soup.find_all(tag):
      headings[tag].append(heading.text.strip())
    
  return headings

def extract_links(html):
  soup = BeautifulSoup(html, 'html.parser')
    
  links = []
  for a_tag in soup.find_all('a'):
      link_info = {
          'text': a_tag.text.strip(),
          'href': a_tag.get('href', ''),
          'title': a_tag.get('title', '')
      }
      links.append(link_info)
    
  return links

def extract_patterns(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()  # Get text for regex operations
    
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phone_count = len(re.findall(phone_pattern, text))
    
    course_pattern = r'[A-Z]{2}\s*\d{3}'
    has_course_number = bool(re.search(course_pattern, text))

    return {'phone_count': phone_count, 'has_course_number': has_course_number}

def extract_objects(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    objects = {
        'paragraph_count': len(soup.find_all('p')),
        'hr_count': len(soup.find_all('hr')), 
        'image_count': len(soup.find_all('img'))
    }
    
    return objects




def extract_keywords(html):
  " we can select a set of words we think would frequently be associated with target classes, and count their occurences in a document "

  soup = BeautifulSoup(html, 'html.parser')
  text = soup.get_text().lower()
  
  keywords = {'student_indicators': {
          'keywords': ['student', 'organization', 'club', 'society', 'undergraduate', 'graduate', 'bs', 'ms'],
          'count': 0},
          'faculty_indicators': {
          'keywords': ['faculty', 'professor', 'research', 'publication', 'department', 'staff', 'academic', 'phd'],
          'count': 0},
          'course_indicators': {
          'keywords': ['course', 'syllabus', 'semester', 'assignment', 'lecture', 'exam', 'class', 'credit'],'count': 0}}
  
  for category, words in keywords.items():
      for keyword in words['keywords']:
          words['count'] += text.count(keyword)
  
  return keywords


def process_html(html, file=""):
    clean_text = clean_html(html)
    title = extract_title(html)
    patterns = extract_patterns(html)
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
      'title': title,
      'headings': {k: len(v) for k, v in headings.items()},
      'num_total_headings': sum(len(v) for v in headings.values()),
      'num_links': len(links),
      'num_out_links': sum(1 for link in links if link['href'].startswith('http')),
      'num_emails': sum(1 for link in links if link['href'].startswith('mailto:')),
      **patterns,  # Fix: spread the patterns
      **objects,   # Fix: spread the objects
      'type_keywords': {k: v['count'] for k, v in keywords.items()}
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
      
      for filename in tqdm(files, desc=f"processing {label} files"):
          if os.path.isfile(os.path.join(subdir_path, filename)):
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

    for filename in tqdm(files, desc=f"processing {label} files"):    # bar to monitor bd parsing progress
      if os.path.isfile(os.path.join(dir, filename)):  
        file_path = os.path.join(dir, filename)
        content = read_files(file_path)

        processed = process_html(content, filename)
        processed['label'] = 'unk'      # unk labels to test, combine features to a dataset
        
        data.append(processed)
        labels.append('unk')
  
  return pd.DataFrame(data), labels


def extract_features(data):
    features = []
    
    for idx, row in data.iterrows():
      feature_dict = {
         # binary features 
        'title': 1 if row['title'] else 0,
        'has_course_number': 1 if row['has_course_number'] else 0,

        #count features, integers
        'h1_count': row['headings']['h1'],
        'h2_count': row['headings']['h2'],
        'h3_count': row['headings']['h3'],
        'total_headings': row['num_total_headings'],
        'link_count': row['num_links'],
        'external_links': row['num_out_links'],
        'email_links': row['num_emails'],
        'phone_count': row['phone_count'],
        
        'html_length': row['html_length'],
        'text_length': row['text_length'],
        
        'paragraph_count': row['paragraph_count'],
        'hr_count': row['hr_count'],
        'image_count': row['image_count'],

         # text features, strings

        'student_keywords': row['type_keywords']['student_indicators'],
        'faculty_keywords': row['type_keywords']['faculty_indicators'],
        'course_keywords': row['type_keywords']['course_indicators']
        }
      features.append(feature_dict)
    
    return pd.DataFrame(features)
  




