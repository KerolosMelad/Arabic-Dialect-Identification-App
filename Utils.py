import pandas as pd
import seaborn as sns
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn import preprocessing
from collections import Counter 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import io
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text  import tokenizer_from_json

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    
with open("1st_encoder", "rb") as f: 
    label_encoder = pickle.load(f) 

with open("2nd_encoder", "rb") as f: 
     OneHotEncoder= pickle.load(f) 
        
        




emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\u2069" 
        u"\u2066"
        u"\u200b"
                      "]+", re.UNICODE)

def regex_manupilation(tweet):
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = re.sub(r'[0-9]+', '', tweet) # delete numbers
    tweet = re.sub(r'&amp;|&quot;|&gt;', '', tweet)
    #tweet =re.sub(r'(.)\1+', r'\1', tweet)
    tweet = re.sub(r'\s*[A-Za-z]+\b', '' , tweet) #<=
    tweet = emoji_pattern.sub('', tweet)  #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    return tweet



arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation

def remove_punctuation(tweet):
    punctuations = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations)
    return tweet.translate(translator)

noise = re.compile(""" ّ| َ| ً| ُ| ٌ| ِ| ٍ| ْ| """, re.VERBOSE)

def unify_letters(tweet):
    tweet = re.sub("[إأٱآا]", "ا", tweet)
    tweet = re.sub("ى", "ي", tweet)
    tweet = re.sub("ؤ", "ء", tweet)
    tweet = re.sub("وو", "و", tweet)
    tweet = re.sub("ئ", "ء", tweet)
    tweet = re.sub("گ", "ك", tweet)

    tweet = re.sub(noise, '', tweet)
    return tweet

list_to_remove = ['هلا', 'دونك', 'يناير','فبراير','مارس','أبريل','مايو','يونيو','يوليو','أغسطس','سبتمبر','أكتوبر','نوفمبر','ديسمبر','جانفي','فيفري','مارس','أفريل','ماي','جوان','جويلية','أوت','كانون','شباط','آذار','نيسان','أيار','حزيران','تموز','آب','أيلول','تشرين','دولار','دينار','ريال','درهم','ليرة','جنيه','قرش','مليم','فلس','هللة','سنتيم','يورو','ين','يوان','شيكل','نيف'] 

stopwords_list = list(set(stopwords.words('arabic')) - set(list_to_remove))

stopwords_list_space =[]

for w in stopwords_list :
    stopwords_list_space.append(unify_letters(w))
    
stopwords_list_space = stopwords_list_space+[" "]

def remove_stopWords_spaces(tweet):
    tweets_tokeinzed = [] 
    for w in nltk.word_tokenize(tweet) :
         if (w not in stopwords_list_space) and len(w)>1 :
                tweets_tokeinzed.append(w)
                
    return ' '.join(tweets_tokeinzed)





def Pre_Process(tweets):    
    clean_tweets_untokenized = [] 
    for tweet in tweets:
        tweet= regex_manupilation(tweet)
        tweet= remove_punctuation(tweet)
        tweet = unify_letters(tweet)
        tweet_untokized = remove_stopWords_spaces(tweet)
        clean_tweets_untokenized.append(tweet_untokized)
    return clean_tweets_untokenized

def Put_In_shape(tweet):
    tweet = Pre_Process(tweet)
    tweet = tokenizer.texts_to_sequences(tweet)
    tweet = pad_sequences(tweet, maxlen=250, padding='post', truncating='post')
    return tweet

def Prediction_In_shape(prediction):
    idx = (-prediction).argsort()[:4]
    countries=label_encoder.inverse_transform(idx)
    accuracies = prediction[idx] *100
    return countries[0],countries[1],countries[2],countries[3] , accuracies[0] , accuracies[1] , accuracies[2] , accuracies[3]
