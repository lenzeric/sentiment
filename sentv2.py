# URL grabbing section
import re
import datetime
import requests

#NLP and map sections
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

# User inputs (up to 5 years)
years_in = int(input("Enter the number of years (ex. 1, 2, 5, or 10): "))
#years_in = 1

# Get current year through datetime module
current_year = datetime.datetime.now().year

# Sample list of words
q_words = ["first", "second", "third", "fourth"]
m_words = ["/02/", "/05/", "/08/", "/10/"]
y_words = [str(current_year), str(current_year - 1), str(current_year - 2)]

# Convert list to a regex pattern (case-insensitive search)
q_pattern = r'\b(?:' + '|'.join(map(re.escape, q_words)) + r')\b'
m_pattern = r'\b(?:' + '|'.join(map(re.escape, m_words)) + r')\b'
y_pattern = r'\b(?:' + '|'.join(map(re.escape, y_words)) + r')\b'

# Sample url (AAPL)
text = "https://www.apple.com/newsroom/2024/02/apple-reports-first-quarter-results/"

# Other urls to try...others from top 5 S&P500 companies: NVDA, MSFT, AMZN, META.
#NVDA's 4th quarter is different than other urls: https://investor.nvidia.com/news/press-release-details/2024/NVIDIA-Announces-Financial-Results-for-Fourth-Quarter-and-Fiscal-2024/
#text = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2025"
#Microsoft's urls have a Q1, Q2, Q3, and Q4.
#text = "https://www.microsoft.com/en-us/Investor/earnings/FY-2025-Q1/press-release-webcast"
#Amazon's urls yield 'status code: 403' meaning access to the requested resource is forbidden.
#text = "https://ir.aboutamazon.com/news-release/news-release-details/2024/Amazon.com-Announces-Third-Quarter-Results/"
#Facebook's urls also yield 'status code: 403' meaning access to the requested resource is forbidden.
#text = "https://investor.fb.com/investor-news/press-release-details/2024/Meta-Reports-Third-Quarter-2024-Results/default.aspx"

# Index of the first occurrence of 'apple'.
apple = text.find('apple')

# Find all matches using findall
q_matches = re.findall(q_pattern, text, re.IGNORECASE)
m_matches = re.findall(m_pattern, text, re.IGNORECASE)
y_matches = re.findall(y_pattern, text, re.IGNORECASE)

# Dictionaries for quarter and month and year
if q_matches[0][0].isupper():
    qtr_dict = {
        "First": ["First", "Second", "Third", "Fourth"],
        "Second": ["Second", "Third", "Fourth", "First"],
        "Third": ["Third", "Fourth", "First", "Second"],
        "Fourth": ["Fourth", "First", "Second", "Third"],
    }

elif q_matches[0][0].islower():
    qtr_dict = {
        "first": ["first", "second", "third", "fourth"],
        "second": ["second", "third", "fourth", "first"],
        "third": ["third", "fourth", "first", "second"],
        "fourth": ["fourth", "first", "second", "third"],
    }
    
mon_dict = {
    "/02/": ["/02/", "/05/", "/08/", "/10/"],
    "/05/": ["/05/", "/08/", "/10/", "/02/"],
    "/08/": ["/08/", "/10/", "/02/", "/05/"],
    "/10/": ["/10/", "/02/", "/05/", "/08/"],
}

# Set up year dictionary for 1, 2, 5, and 10 years according to historical stock prices from yfinance's ticker.history(period="")
y_dict = {
    1: [y_matches[0]],
    2: [y_matches[0], str(int(y_matches[0]) - 1)],
    5: [y_matches[0], str(int(y_matches[0]) - 1), str(int(y_matches[0]) - 2), str(int(y_matches[0]) - 3), str(int(y_matches[0]) - 4)],
    10: [y_matches[0], str(int(y_matches[0]) - 1), str(int(y_matches[0]) - 2), str(int(y_matches[0]) - 3), str(int(y_matches[0]) - 4), str(int(y_matches[0]) - 5), str(int(y_matches[0]) - 6), str(int(y_matches[0]) - 7), str(int(y_matches[0]) - 8),str(int(y_matches[0]) - 9)],
}

url_list = []
text_m = []
for y in y_dict[years_in]:
    text_y = text.replace(y_matches[0],y)
    for q in qtr_dict[q_matches[0]]: #if condition for Qs
        text_q = text_y.replace(q_matches[0], q)
        qindex = qtr_dict[q_matches[0]].index(q)
        if m_matches != []:
            text_m = text_q.replace(m_matches[0], mon_dict[m_matches[0]][qindex])
            try:
                response = requests.get(text_m)
                if response.status_code == 200:
                    #print(f"URL is good: {text_m}")
                    url_list.append(text_m)
                else:
                    #print(f"URL is not good (status code: {response.status_code}): {text_m}")
                    plus_1 = '/'+str(int(mon_dict[m_matches[0]][qindex].strip("/"))+1).zfill(2)+'/'
                    text_m = text_m.replace(mon_dict[m_matches[0]][qindex], plus_1)
                    response = requests.get(text_m)
                    if response.status_code == 200:
                        #print(f"URL is good: {text_m}")
                        url_list.append(text_m)
                    else:
                        #print(f"URL is not good (status code: {response.status_code}): {text_m}")
                        minus_1 = '/'+str(int(plus_1.strip("/"))-2).zfill(2)+'/'
                        text_m = text_m.replace(plus_1, minus_1)
                        response = requests.get(text_m)
                        if response.status_code == 200:
                            #print(f"URL is good: {text_m}")
                            url_list.append(text_m)
                        else:
                            print(f"URL is not good (status code: {response.status_code}): {text_m}")
            except requests.exceptions.RequestException as e:
                print(f"Error with URL {text_m}: {e}")
        else:
            response = requests.get(text_q)
            if response.status_code == 200:
                print(f"URL is good: {text_q}")
                url_list.append(text_q)
            else:
                print(f"URL is not good (status code: {response.status_code}): {text_m}")
                            
print("You have "+str(len(url_list))+" quarterly reports.")

# NLP section.
data = []
siascores = []
vaderscores = []
tblobscores = []
tblobscores2 = []
afinnscores = []
dates = []
for url in url_list :
    text = " "
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    #view page source for details on NVDA.
    if apple != -1:
        text_all = soup.find_all(attrs={"class": "pagebody-copy"})
        for element in text_all:
            text_0 = element.get_text(" ", strip=True)
            text = text + text_0
    else:
        body = soup.body
        p_tags = body.find_all('p', limit=6) #adjust limit
        for p in p_tags:
            text_0 = p.text
            #print(text_0)
            text = text + text_0
        
    sent = sent_tokenize(text)
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
    
    #Use datefinder to find first match in text. Then break because date always at beginning of quarterly release.
    matches = datefinder.find_dates(unique_string_v2)
    for match in matches:
        break
    dates.append(match)
    
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #pip install vaderSentiment
    vader = SentimentIntensityAnalyzer()
    vsent = vader.polarity_scores(unique_string_v2)

    from textblob import TextBlob #pip install textblob
    blob = TextBlob(unique_string_v2)
    bsent = blob.sentiment
    
    from afinn import Afinn #pip install Afinn
    afinn = Afinn()
    asent = afinn.score(unique_string_v2)

    #Create dataframes from two sentiment intensity analyzers, concat, and add new columns of scores for other two sentiment intensity analyzers.
    siascores.append(sia.polarity_scores(unique_string_v2))
    vaderscores.append(vsent)
    tblobscores.append(bsent.polarity)
    tblobscores2.append(bsent.subjectivity)
    afinnscores.append(asent)

df1 = pd.DataFrame(siascores)
df1.columns = ['sia_neg', 'sia_neu', 'sia_pos', 'sia_compound']

dfvader = pd.DataFrame(vaderscores)
dfvader.columns = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
df1 = pd.concat([df1, dfvader], axis=1)

df1['Afinn'] = afinnscores
df1['tblob_polarity']= tblobscores
df1['tblob_subjectivity']= tblobscores2

#Append two lists of text and sentiment scores.
data.append(unique_string_v2)

#Insert the list of dates into the first column (0) and call it Date.
df1.insert(0,'Date', dates)

#Sort the data by Date and print.
df1 = df1.sort_values(by='Date')
print(df1.head(10))


# yfinance section
#Define the ticker symbol and create a Ticker object.
ticker_symbol = "AAPL"
ticker = yf.Ticker(ticker_symbol)

#Fetch historical market data and make dataframe df2. Period must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'] 
historical_data = ticker.history(period=str(years_in)+"y")
df2 = pd.DataFrame(historical_data)

#Make a new datetime index for df1 to match df2 columns for a plot.
df1['time'] = '00:00:00-05:00'
df1['Date'] = df1['Date'].astype(str)
df1['Datetime'] = pd.to_datetime(df1['Date'] + ' ' + df1['time'], format='%Y-%m-%d %H:%M:%S%z')
df1 = df1.set_index(pd.DatetimeIndex(df1['Datetime']))

# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# First plot with two y-axes
ax1 = axes[0]
ax1.plot(df2.index, df2['Close'], label="Closing price", color="green")
ax1.set_ylabel("Closing price", color="green")
ax1.tick_params(axis='y', labelcolor="green")
ax1.set_title("TextBlob and Afinn sentiment scores with AAPL closing price")

# Create second y-axis for the first plot
ax2 = ax1.twinx()
ax2.plot(df1.index, df1['tblob_polarity'], label="tblob Sentiment Index", color="red")
ax2.set_ylabel("tblob Sentiment Index", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# Add legends to avoid overlap
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Second plot with two y-axes
ax3 = axes[1]
ax3.plot(df2.index, df2['Close'], label="Closing price", color="green")
ax3.set_ylabel("Closing price", color="green")
ax3.tick_params(axis='y', labelcolor="green")
ax3.set_xlabel("Date")

# Create second y-axis for the second plot
ax4 = ax3.twinx()
ax4.plot(df1.index, df1['Afinn'], label="Afinn", color="purple")
ax4.set_ylabel("Afinn", color="purple")
ax4.tick_params(axis='y', labelcolor="purple")

# Add legends to avoid overlap
ax3.legend(loc="upper left")
ax4.legend(loc="upper right")

# Adjust layout
plt.tight_layout()
plt.show()



# Create subplots with 2 rows and 1 column
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# First plot with two y-axes
ax1 = axes[0]
ax1.plot(df2.index, df2['Close'], label="Closing price", color="green")
ax1.set_ylabel("Closing price", color="green")
ax1.tick_params(axis='y', labelcolor="green")
ax1.set_title("SIA and VADER sentiment scores with AAPL closing price")

# Create second y-axis for the first plot
ax2 = ax1.twinx()
ax2.plot(df1.index, df1['sia_compound'], label="SIA Sentiment Index", color="red")
ax2.set_ylabel("SIA Sentiment Index", color="red")
ax2.tick_params(axis='y', labelcolor="red")

# Add legends to avoid overlap
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Second plot with two y-axes
ax3 = axes[1]
ax3.plot(df2.index, df2['Close'], label="Closing price", color="green")
ax3.set_ylabel("Closing price", color="green")
ax3.tick_params(axis='y', labelcolor="green")
ax3.set_xlabel("Date")

# Create second y-axis for the second plot
ax4 = ax3.twinx()
ax4.plot(df1.index, df1['vader_compound'], label="VADER Sentiment Index", color="purple")
ax4.set_ylabel("VADER Sentiment Index", color="purple")
ax4.tick_params(axis='y', labelcolor="purple")

# Add legends to avoid overlap
ax3.legend(loc="upper left")
ax4.legend(loc="upper right")

# Adjust layout
plt.tight_layout()
plt.show()

#pip install pandas openpyxl
# Ask the user if they want to export to Excel
user_input = input("Do you want to export to Excel? (y/n): ").strip().lower()

#Make a column of strings for dates in df2. For export to Excel.
df2['date_strings'] = df2.index.strftime('%Y-%m-%d')
df1['date_strings'] = df1.index.strftime('%Y-%m-%d')

#df1 and df2 share the same datetime index, pd.concat can combine them directly. Then reset the datetime index to remove the timezone and export to Excel.
merged = pd.concat([df1, df2], axis=1)
merged['datestrings'] = df1['date_strings'].combine_first(df2['date_strings'])
merged = merged.drop(["Date", "Datetime", "date_strings", "time"], axis=1)
merged.insert(0, 'Date', merged.pop('datestrings'))
print(merged.info())

if user_input == "y":
    print("Exporting to Excel...")
    merged.to_excel('output.xlsx', index=False)
elif user_input == "n":
    print("Skipping Excel export.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")

#Setup for correlation coeff...still requires work, interpolate sentiment scores for matching x and y # of obs.
#from scipy.stats import pearsonr #pip install scipy
#corr, _ = pearsonr(df1['tblob_polarity'], df2['Close'])
#plt.text(0.05, 0.95, f"Pearson r = {corr:.2f}", 
#         transform=plt.gca().transAxes, fontsize=12, 
#         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
#plt.show()