from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd #pip install pandas
import datefinder #pip install datefinder
import matplotlib.pyplot as plt #pip install matplotlib
import matplotlib.dates as mdates 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

url_list = ['https://www.govinfo.gov/content/pkg/CREC-2024-03-07/html/CREC-2024-03-07-pt1-PgS2272-2.htm', 'https://www.govinfo.gov/content/pkg/DCPD-202300096/html/DCPD-202300096.htm', 'https://www.govinfo.gov/content/pkg/DCPD-202200127/html/DCPD-202200127.htm', 'https://www.govinfo.gov/content/pkg/DCPD-202100347/html/DCPD-202100347.htm', 'https://www.govinfo.gov/content/pkg/DCPD-202000058/html/DCPD-202000058.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201900063/html/DCPD-201900063.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201800064/html/DCPD-201800064.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201700150/html/DCPD-201700150.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201600012/html/DCPD-201600012.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201500036/html/DCPD-201500036.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201400050/html/DCPD-201400050.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201300090/html/DCPD-201300090.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201200048/html/DCPD-201200048.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201100047/html/DCPD-201100047.htm', 'https://www.govinfo.gov/content/pkg/DCPD-201000055/html/DCPD-201000055.htm', 'https://www.govinfo.gov/content/pkg/DCPD-200900105/html/DCPD-200900105.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2008-02-04/html/WCPD-2008-02-04-Pg117.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2007-01-29/html/WCPD-2007-01-29-Pg57.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2006-02-06/html/WCPD-2006-02-06-Pg145-3.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2005-02-07/html/WCPD-2005-02-07-Pg126.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2004-01-26/html/WCPD-2004-01-26-Pg94-2.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2003-02-03/html/WCPD-2003-02-03-Pg109.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2002-02-04/html/WCPD-2002-02-04-Pg133-3.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2001-03-05/html/WCPD-2001-03-05-Pg351-2.htm', 'https://www.govinfo.gov/content/pkg/WCPD-2000-01-31/html/WCPD-2000-01-31-Pg160-2.htm', 'https://www.govinfo.gov/content/pkg/WCPD-1999-01-25/html/WCPD-1999-01-25-Pg78-2.htm', 'https://www.govinfo.gov/content/pkg/WCPD-1998-02-02/html/WCPD-1998-02-02-Pg129-2.htm', 'https://www.govinfo.gov/content/pkg/WCPD-1997-02-10/html/WCPD-1997-02-10-Pg136.htm', 'https://www.govinfo.gov/content/pkg/WCPD-1996-01-29/html/WCPD-1996-01-29-Pg90.htm', 'https://www.govinfo.gov/content/pkg/WCPD-1995-01-30/html/WCPD-1995-01-30-Pg96.htm', 'https://www.govinfo.gov/content/pkg/WCPD-1994-01-31/html/WCPD-1994-01-31-Pg148.htm', 'https://www.govinfo.gov/content/pkg/WCPD-1993-02-22/html/WCPD-1993-02-22-Pg215-2.htm']

# NLP section.
data = []
siascores = []
vaderscores = []
tblobscores = []
tblobscores2 = []
afinnscores = []
dates = []
year = 2025
year_list = []
for url in url_list :
    text = " "
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
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

    siascores.append(sia.polarity_scores(unique_string_v2))
    vaderscores.append(vsent)
    tblobscores.append(bsent.polarity)
    tblobscores2.append(bsent.subjectivity)
    afinnscores.append(asent)
    
    year -=1
    year_list.append(year)

df1 = pd.DataFrame(siascores)
df1.columns = ['sia_neg', 'sia_neu', 'sia_pos', 'sia_compound']

dfvader = pd.DataFrame(vaderscores)
dfvader.columns = ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
df1 = pd.concat([df1, dfvader], axis=1)

df1['Afinn'] = afinnscores
df1['tblob_polarity']= tblobscores
df1['tblob_subjectivity']= tblobscores2
df1.insert(0,'Date', year_list)

#Append two lists of text and sentiment scores.
data.append(unique_string_v2)

# Link to my Google sheet of IMF World Economic Outlook (WEO) data for US economic growth
#https://docs.google.com/spreadsheets/d/1NAIKLv1PddRpzczkTYNMjkexeknkbC-Ecp_FsJtNSiw/edit?usp=sharing

# Identify my Sheet ID and Sheet Name from previous link
sheet_id = "1NAIKLv1PddRpzczkTYNMjkexeknkbC-Ecp_FsJtNSiw"
sheet_name = "Sheet1"  # Replace with the specific sheet name if needed

# Construct the URL for CSV export
gsheet = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# Read the Google Sheet into a DataFrame
df2 = pd.read_csv(gsheet)
df1 = pd.concat([df1, df2], axis=1)

# Check the concatenated df to make sure Date and Year line up correctly.
#print(df1.head(10))

# Convert the 'date' column to datetime type to easily lag growth variable with shift function of dataframe module.
df1['Date'] = pd.to_datetime(df1['Date'], format='%Y')

# Sort the DataFrame by the 'date' column
df1 = df1.sort_values(by='Date')

# Create a lagged variable
df1['lagged_growth'] = df1['Growth'].shift(1)
#print(df1.head(10))

# Show dataframe info
#print(df1.info())

# Standardizing the dataframe for an average sentiment score
df_standardized = (df1 - df1.mean()) / df1.std()

# 'sia_compound' and 'vader_compound' do a very poor job at analyzing the sentiment of the addresses with very little variation after the economic recovery from the great financial crisis. Consider a plot with those columns over time for verification.
#df1['std_sent'] = (df_standardized['Afinn'] + df_standardized['tblob_polarity'] + df_standardized['sia_compound'] + df_standardized['vader_compound'])/4

# Average sentiment score from Afinn and Tblob
df1['std_sent'] = (df_standardized['Afinn'] + df_standardized['tblob_polarity'])/2

#Plot with two y-axes.
col1 = 'black'
col2 = 'red'
fig,ax = plt.subplots()
ax.plot(df1['Year'], df1['lagged_growth'], color=col1)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Growth, t-1', color=col1, fontsize=16)
ax2 = ax.twinx()
ax2.plot(df1['Year'], df1['std_sent'], color=col2)
ax2.set_ylabel('Standardized Sentiment Index', color=col2, fontsize=16)
ax.set_title("State of Union Address Sentiment Scores and Economic Growth in the Previous Year")

# Shading for republican/democratic administrations
ax.axvspan(1993, 2000, color='blue', alpha=0.1)
ax.axvspan(2000, 2008, color='red', alpha=0.1)
ax.axvspan(2008, 2016, color='blue', alpha=0.1)
ax.axvspan(2016, 2020, color='red', alpha=0.1)
ax.axvspan(2020, 2024, color='blue', alpha=0.1)
plt.show()