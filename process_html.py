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

def extract_metadata(html):

  soup = BeautifulSoup(html, 'html.parser')
    
  meta_info = {'title': soup.title.text if soup.title else '','description': '','keywords': '','author': ''}
    
  for meta in soup.find_all('meta'):
      name = meta.get('name', '').lower()
      content = meta.get('content', '')
      
      if name == 'description':
          meta_info['description'] = content
      elif name == 'keywords':
          meta_info['keywords'] = content
      elif name == 'author':
          meta_info['author'] = content
  
  return meta_info


def extract_objects(html):
  soup = BeautifulSoup(html, 'html.parser')
    
  objects = {
    'header': len(soup.find_all('header')) > 0,
    'nav': len(soup.find_all('nav')) > 0,
    'main': len(soup.find_all('main')) > 0,
    'footer': len(soup.find_all('footer')) > 0,
    'aside': len(soup.find_all('aside')) > 0,
    'articles': len(soup.find_all('article')),
    'sections': len(soup.find_all('section'))
}
    
  return objects

def extract_keywords(html):
  " we can select a set of words we think would frequently be associated with target classes, and count their occurences in a document "

  soup = BeautifulSoup(html, 'html.parser')
  text = soup.get_text().lower()
  
  keywords = {'student_indicators': {
          'keywords': ['student', 'students', 'organization', 'club', 'society', 'undergraduate', 'graduate'],
          'count': 0},
          'faculty_indicators': {
          'keywords': ['faculty', 'professor', 'research', 'publication', 'department', 'staff', 'academic'],
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

  for idx, row in data.iterrows():

    feature_dict = {
      'title': 1 if row['title'] else 0,
      'description': 1 if row['description'] else 0,
      'header': 1 if row['has_header'] else 0,
      'footer': 1 if row['has_footer'] else 0,
      'main': 1 if row['has_main'] else 0,
      'nav': 1 if row['has_nav'] else 0,
      'keywords': 1 if row['keywords'] else 0,
      'len_txt': row['text_length'],
      'len_hmtl': row['html_length'],
      'count_h1': row['headings']['h1'],
      'count_h2': row['headings']['h2'],
      'count_h3': row['headings']['h3'],
      'count_headings': row['num_total_headings'],
      'count_links': row['num_links'],
      'count_sections': row['num_sections'],
      'count_student_kws': row['type_keywords']['student_indicators'],
      'count_faculty_kws': row['type_keywords']['faculty_indicators'],
      'count_course_kws': row['type_keywords']['course_indicators']}

    features.append(feature_dict)


  return pd.DataFrame(features)
  




