'''
sandipan
ghoshm21@gmail.com
29th Aug 2020
'''
from bs4 import BeautifulSoup
import unidecode
import spacy
import gensim.downloader as api
from pycontractions import Contractions
import gensim.downloader as api
import gensim
import os
import pandas as pd
import re
import string
from glob import glob
import gc

'''-----------------------------------------------------------------------------------------------------------'''
''' read all input data'''
# read all the input data files, tab seperated values
data_txt = glob('/home/sandipan/Documents/deep_learning/sentiment_analysis/data/*.txt')
data_tsv = glob('/home/sandipan/Documents/deep_learning/sentiment_analysis/data/*.tsv')
model_file = '/tf/deep_learning/sentiment_analysis/lib/model'
header_list = ["comments", "sentiment"]
# read all the data using windows encoding and python engine. Else it will give error for windows files
# l = [pd.read_csv(f, sep='\t', names=header_list, encoding = "ISO-8859-1", engine='python') for f in data_txt]
l = [pd.read_csv(f, sep='\t', names=header_list) for f in data_txt]
data_txt = pd.concat(l, axis=0)
print('total length of the training data_txt %s'%(len(data_txt)))

# read all the data using windows encoding and python engine. Else it will give error for windows files
# l = [pd.read_csv(f, sep='|', names=header_list, encoding = "ISO-8859-1", engine='python') for f in data_tsv]
l = [pd.read_csv(f, sep='|', names=header_list) for f in data_tsv]
data_tsv = pd.concat(l, axis=0)
print('total length of the training data_tsv %s'%(len(data_tsv)))
# total data
frames = [data_txt, data_tsv]
data_raw = pd.concat(frames)
print('total length of the training data %s'%(len(data_raw)))

# drop the base data and free the memory
del [[l,data_txt, data_tsv]]
gc.collect()
'''-----------------------------------------------------------------------------------------------------------'''

def remove_html(text):
  """remove html tags from text"""
  soup = BeautifulSoup(text, "html.parser")
  stripped_text = soup.get_text(separator=" ")
  return stripped_text

def remove_accented_chars(text):
  """remove accented characters from text, e.g. café"""
  text = unidecode.unidecode(text)
  return text
    
def remove_tabs(text):
  '''remove all the tab, new line char'''
  text = text.replace('\t', ' ')
  text = text.replace('\r', ' ')
  text = text.replace('\n', ' ')
  return text

def remove_blanks(text):
  '''remove all the more than 1 spaces'''
  text = re.sub(' +', ' ', text)
  return text

def remove_digits(text):
  # Remove digits, decimal numbers, dates and time format
  return re.sub(r'\d[\.\/\-\:]\d|\d', '', text)
    
def remove_all_punctuation(text):
  '''Remove other punctuation, adding fe more
  string.punctuation = !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`
  '''
  PUNCT_TO_REMOVE = string.punctuation
  return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_special_characters(text, remove_digits=False):
  pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
  text = re.sub(pattern, '', text)
  return text


def ascii_to_string(text):
  # Encodes string to ASCII and decodes to string. This helps in removing any special characters in the database
  text = text.encode('ascii', 'replace').decode(encoding="utf-8")
  '''
  This replaces all special characters with a ?. Replacing this
  '''
  return text.replace('?', '')


def to_lower(text):
  '''conver all to lower'''
  return str(text).lower( )

def remove_url(text):
  # Remove any web url starting with http or www
  return re.sub(r'(www|http)\S+', '', text)

def remove_email_address(text):
  # Remove any email address
  return re.sub(r'\S+@\S+', '', text)

def expand_contractions(text):
  """expand shortened words, e.g. don't to do not"""
  text = list(cont.expand_texts([text], precise=True))[0]
  return text
  
nlp = spacy.load('en_core_web_md')
#api.BASE_DIR = os.path.abspath('/home/sandipan/Documents/deep_learning/sentiment_analysis/')
model = api.load('word2vec-google-news-300')
print("The word2vec is present in: " + api.load("word2vec-google-news-300", return_path=True))
cont = Contractions(kv_model=model)
cont.load_models()

# load data
#data_raw = pd.read_csv('/home/sandipan/Documents/deep_learning/sentiment_analysis/test_data.text')
#pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", -1)
print(" ------------------- raw data -------------------")
print(data_raw)
#print(list(cont.expand_texts(["I'd like to know how I'd done that!",
#                            "We're going to the zoo and I don't think I'll be home for dinner.",
#                            "Theyre going to the zoo and she'll be home for dinner."], precise=True)))

# apply the Expand Contractions
data_raw['comments'] = data_raw['comments'].apply(remove_html).apply(remove_accented_chars).apply(remove_tabs).apply(remove_blanks).apply(remove_digits).apply(ascii_to_string).apply(remove_special_characters).apply(remove_url).apply(remove_email_address).apply(expand_contractions).apply(to_lower).apply(remove_all_punctuation)
print(" ------------------- clean data -------------------")
print(data_raw)
data_raw.to_csv("/home/sandipan/Documents/deep_learning/sentiment_analysis/data/sentiment_all_clean.csv")
