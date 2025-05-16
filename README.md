# Data Mining Project [course id]

This repo contains the code, data, notebooks, and reports for the final project of the course Introduction to Data mining. The project consists of two parts: 1. data exploration and 2. html website data classification. see ./project.pdf

## Part 1. Physical Activity Data Exploration

The report on findings in the activities dataset can be find in report.pdf. The notebook with data manipulations can be found at ./part1/activities.ipynb

## Part 2. Website Classification

The report on the classifiers is in the same combined report.pdf. The project can easily be reproduced within dm-project/ directory. This part of the project classifies html source code of web pages into three categories:

- student
- faculty
- course

We implemented a simple text classification pipeline that trains and evaluates a variety of models and feature sets. Our setup contains:

1. preprocessing html source code
2. extracting features
3. training classifiers
4. making predictions on the test data
5. evaluating and comparing models

### Installation and execution

- Python >= 3.9+, scikit-learn, pandas, numpy, beautifulsoup4

> see `environment.yml` and `requirements.txt`


#### installation

#### running the analysis

```python
# run the command below within the project dir
python main.py --train_dir data/train --test_dir data/test
```

the script should also load these directories by default, so the setup should work just as well with no argparse flag, ie python main.py
