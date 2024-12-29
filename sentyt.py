#AAPL quarterly report sentiment analysis with natural language programming.
#Author: Eric Lenz, PhD, CBE

#Description
#This program scrapes text from four AAPL quarterly reports and performs sentiment analysis using NLTK's SentimentIntensityAnalyzer. The sentiment of the text is measured by positive, negative, or neutral scores in addition to a compound score from -1 to +1. I then compare the compound scores to historical AAPL price data found on Yahoo with yfinance in a plot.

#Code is my own or sourced from the following.
#format multiple strings: https://stackoverflow.com/questions/1225637/inserting-the-same-value-multiple-times-when-formatting-a-string
#Use .find_all() instead of .find(): https://scrapeops.io/python-web-scraping-playbook/python-beautifulsoup-findall/
#Datefinder: https://datefinder.readthedocs.io/en/latest/
#More with Datefinder: https://pythonguides.com/add-elements-in-list-in-python-using-for-loop/
#Plotting w/ two y-axes: https://www.statology.org/matplotlib-two-y-axes/

#Apple quarterly reports (press releases)
#https://www.apple.com/newsroom/2024/02/apple-reports-first-quarter-results/
#https://www.apple.com/newsroom/2024/05/apple-reports-second-quarter-results/
#https://www.apple.com/ml/newsroom/2024/08/apple-reports-third-quarter-results/
#https://www.apple.com/newsroom/2024/10/apple-reports-fourth-quarter-results/

import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd #pip install pandas
import datefinder #pip install datefinder
import yfinance as yf #pip install yfinance
import matplotlib.pyplot as plt #pip install matplotlib
import matplotlib.dates as mdates 

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

quarter = ['first', 'second', 'third', 'fourth']
month = ['02', '05', '08', '10']

data = []
scores = []
dates = []
for releases in range(len(quarter)) :
    text = " "
    url = 'https://www.apple.com/newsroom/2024/{a}/apple-reports-{b}-quarter-results/'.format(a=month[releases], b=quarter[releases])
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read()
    soup = BeautifulSoup(html, 'html.parser')
    text_all = soup.find_all(attrs={"class": "pagebody-copy"})
    #print(text_all)
    for element in text_all:
        text_0 = element.get_text(" ", strip=True)
        #print(text_0)
        text = text + text_0
        
    sent = sent_tokenize(text)
    #print(sent)
    words = [word_tokenize(t) for t in sent]
    list_words = sum(words,[])
    low_words = [w.lower() for w in list_words]
    remove_words = [w for w in low_words if w not in stopwords.words('english')]
    punc_words = [w for w in remove_words if w.isalnum()]

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    final_words = [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in punc_words]
    delete_lst = ['cupertino', 'california']
    words_v2 = [w for w in final_words if w not in delete_lst]
    unique_string_v2=(" ").join(words_v2)
    #print(unique_string_v2)
    
    #Use datefinder to find first match in text. Then break because date always at beginning of quarterly release.
    matches = datefinder.find_dates(unique_string_v2)
    for match in matches:
        #print(match)
        break
    dates.append(match)
    
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    #print(sia.polarity_scores(unique_string_v2))
    
    #Append two lists of text and sentiment scores.
    data.append(unique_string_v2)
    scores.append(sia.polarity_scores(unique_string_v2))

#Create the DataFrame: 
df1 = pd.DataFrame(scores)

#Insert the list of dates into the first column (0) and call it Date.
df1.insert(0,'Date', dates)
print(df1.head(10))

#Define the ticker symbol and create a Ticker object.
ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)

#Fetch historical market data and make dataframe df2.
historical_data = ticker.history(period="1y")
df2 = pd.DataFrame(historical_data)

#print(df2.head(10))
#print(df2.info())

#Make a new datetime index for df1 to match df2 columns for a plot.
df1['time'] = '00:00:00-05:00'
df1['Date'] = df1['Date'].astype(str)
df1['Datetime'] = pd.to_datetime(df1['Date'] + ' ' + df1['time'], format='%Y-%m-%d %H:%M:%S%z')
df1 = df1.set_index(pd.DatetimeIndex(df1['Datetime']))
#print(df1.head(10))

#Plot with two y-axes.
col1 = 'steelblue'
col2 = 'red'
fig,ax = plt.subplots()
ax.plot(df2.index, df2['Close'], color=col1)
ax.set_xlabel('Day', fontsize=14)
ax.set_ylabel('Closing Price', color=col1, fontsize=16)
ax2 = ax.twinx()
ax2.plot(df1.index, df1['compound'], color=col2)
ax2.set_ylabel('Sentiment Index', color=col2, fontsize=16)
plt.show()
