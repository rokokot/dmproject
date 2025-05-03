import os
import argparse
import pandas as pd
from process_html import load_datasets
from extract_features import ##
from evaluate_model import *

"""
Main experiment runner for the pipeline, 

Args:
- train dir
- test dir


"""

def main():
  parser = argparse.ArgumentParser(description='Intro to Data Mining Project, Webpage (html) classification')

  parser.add_argument('--train_dir', default='./data/train', help='enter path to train data')
  parser.add_argument('--test_data', default='./data/test', help='enter path to test data')
  args = parser.parse_args()

  os.makedirs('results', exist_ok=True)





if __name__ == '__main__':
  main()
  