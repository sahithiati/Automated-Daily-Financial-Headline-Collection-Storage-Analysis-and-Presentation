import streamlit as st
import numpy as np
import matplotlib as plt
from pymongo import MongoClient
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from newspaper import Article
import nltk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title('  Automated Daily Financial News Analysis Dashboard')
#connect to MongoDB
client = MongoClient('mongodb+srv://team:JtRbTot4ERYO4acZ@cluster0.qewkk2p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
#db=client['Data_Retrieve']
#collection = db['Unique News Headlines']
db=client['Capstone']
collection = db['Headlines with Description']

# Get today's date and yesterday's date
today = datetime.today().strftime('%Y-%m-%d')
yesterday = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
yesterday_1 = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
# Date intervals to be retrieved
start_date= st.text_input('Enter Start Date', yesterday)
end_date= st.text_input('Enter End Date', today)
# Query the data
cursor = collection.find({"Published Date" : {"$gte": start_date, "$lt": end_date}})
# Convert cursor to list of dictionaries
data_list = list(cursor)
# Check if any data was retrieved
if data_list:
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    print("\nDataFrame created successfully. Shape:", df.shape)
    # Print the first few rows of the DataFrame
    print("\nFirst few rows of DataFrame:")
    print(df.head())
else:
    print("No data found matching the query.")
#Sentimental Analysis using Vader 
vader = SentimentIntensityAnalyzer()
f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['Headline'].apply(f)
df.head()

# Title
st.subheader('Sentimental Score of News')
plt.figure(figsize=(20,20))
mean_df = df.groupby(['Published Date','Stock Identifier']).mean().unstack()
mean_df = mean_df.xs('compound', axis="columns")
mean_df.plot(kind='bar')
#fig=plt.show()
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.pyplot(fig)
plt.xlabel('Published Date')
plt.ylabel('Mean Sentiment Score')
plt.title('Mean Sentiment Score of News by Published Date')
plt.legend(title='Stock Identifier', loc='upper left')
plt.xticks(rotation=45)
st.set_option('deprecation.showPyplotGlobalUse', False)
fig=plt.tight_layout()
# Display the plot in Streamlit
st.pyplot(fig)

# Title
st.title('Stock Price Visualization')
# Get user input for stock symbol
user_input = st.text_input('Enter stock ticker symbol (e.g., AAPL):', 'AAPL')
# Define start and end dates
#start_date = '2022-01-01'
#end_date = '2022-12-31'
# Get stock data from Yahoo Finance
stock_data = yf.download(user_input, start=start_date, end=end_date)
# Plot Open, High, Low, Close prices
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stock_data['Open'], label='Open', color='blue')
ax.plot(stock_data['High'], label='High', color='green')
ax.plot(stock_data['Low'], label='Low', color='red')
ax.plot(stock_data['Close'], label='Close', color='purple')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'{user_input} Stock Price')
ax.legend()
# Display the plot in Streamlit
st.pyplot(fig)

# Title                   
st.subheader('Summarize News Article')
#connect to MongoDB
client = MongoClient('mongodb+srv://team:JtRbTot4ERYO4acZ@cluster0.qewkk2p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db=client['Capstone']
collection = db['Headlines with Description']
# Date intervals to be retrieved
stock= st.text_input('Enter ticker symbol to summarize news', 'AAPL')
# Date intervals to be retrieved
target_date = st.text_input('Enter Date', yesterday)
# Query the data
cursor = collection.find({"Published Date" : target_date, "Stock Identifier" : stock })
# Convert cursor to list of dictionaries
data_list = list(cursor)
# Check if any data was retrieved
if data_list:
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    print("\nDataFrame created successfully. Shape:", df.shape)
    # Print the first few rows of the DataFrame
    print("\nFirst few rows of DataFrame:")
    print(df.head())
else:
    print("No data found matching the query.")  
Links= df['Link']
for link in Links:
    try:
        article = Article(link)
        article.download()
        article.parse()
        article.nlp()
        summarized_output = article.summary 
        summary=f'Summary for {link}: {summarized_output}'
        st.write(summary)
    except ArticleException as e:
        st.warning(f"Failed to download article: {e}")





