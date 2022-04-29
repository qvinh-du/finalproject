"""
References

Machine Learning to Predict Stock Prices:
https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233 

Twitter Sentiment Analysis using Python
https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/ 

Streamlit 101: An in-depth introduction:
https://towardsdatascience.com/streamlit-101-an-in-depth-introduction-fc8aad9492f2 
"""


#Import packages and libraries

#Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date
import math
import os.path
from PIL import Image

#Finance
import yfinance as yf

#Modelling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM 

#Twitter and NLP
import tweepy #need to pip install first
import preprocessor as preprocess #need to pip install first
import re
from textblob import TextBlob #need to pip install first
import nltk
nltk.download('punkt')

#Web
import streamlit as st
from plotly import graph_objs as go

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Twitter API Keys
consumer_key= 'r4G4jn1kjUiMCSzr7rpmyz1Yv'
consumer_secret= 'i4sAmLzvethIHISYWUu8gricaQ7F2uyw7LitKOihFo1KTidFt5'
access_token='1505192442605314057-Ehu1ltCoGVlpRQhnmktFV6IGvKP6Ti'
access_token_secret='5FCsWKq2WZ2ZMQLt9MOF1OXYqvchdwqYb67DmgGFGDbRP'

#Data fetch function
def get_quote(ticker):
    """
    Function to check if our ticker CSV exists. If not, it will get our stock ticker data via Yahoo Finance API
    It will filter into a panda.Dataframe with the relevant informations and store into a CSV file.
    It will then return the CSV file path and the ticker's company name
    """
    
    info_filename = info_filename = 'tickerinfo/'+ ticker + str(date.today()) +'.csv'
    ticker_name = yf.Ticker(ticker).info['shortName']
    
    #Detect if a model file is present
    if (os.path.exists(info_filename) == False):
        end = date.today()
        start = end - datetime.timedelta(days=2 * 365)
        data = yf.download(ticker, start=start, end=end)
        df = pd.DataFrame(data = data)
        df.to_csv(info_filename)
        
    return info_filename, ticker_name

#Price prediction algorithm function
def predict_price(df, ticker):
    """
    Function which will analyze the chosen ticker and its DataFrame as inputs.
    It will return the next day's predicted price and the RMSE error between
    the real and predicted values by the model as the file path for
    image file of the real vs predicted price plot
    """
    #Split data into training set and test dataset
    train_ds = df.iloc[0:int(0.8*len(df)),:]
    test_ds = df.iloc[int(0.8*len(df)):,:]
    
    prediction_days = 7

    training_set=df.iloc[:,4:5].values

    #Scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = scaler.fit_transform(training_set)

    x_train=[]
    y_train=[]
    
    for i in range(prediction_days,len(training_set_scaled)):
        x_train.append(training_set_scaled[i-prediction_days:i,0])
        y_train.append(training_set_scaled[i,0])
        
    #Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    X_forecast = np.array(x_train[-1,1:])
    X_forecast = np.append(X_forecast,y_train[-1])
    
    #Reshaping: Adding 3rd dimension
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))#.shape 0=row,1=col
    X_forecast = np.reshape(X_forecast, (1,X_forecast.shape[0],1))

    filename = 'modelh5/' + str(ticker)+'_model.h5'
    
    #Detect if a model file is present
    if (os.path.exists(filename)):
        model = load_model(filename)
    else:
    
        #Initialise RNN
        model = Sequential()

        #Add first LSTM layer
        model.add(LSTM(units = 50,return_sequences=True,input_shape=(x_train.shape[1],1)))
        model.add(Dropout(0.3))

        model.add(LSTM(units = 75,return_sequences=True))
        model.add(Dropout(0.4))

        model.add(LSTM(units = 100,return_sequences=True))
        model.add(Dropout(0.5))

        model.add(LSTM(units = 125))
        model.add(Dropout(0.6))

        model.add(Dense(units = 1))

        #Compile
        model.compile(optimizer='adam',loss='mean_squared_error')

        #Training
        model.fit(x_train, y_train, epochs = 50, batch_size = 32 )
        
        #Saving model for this specific ticker
        model.save(filename)

    #Testing
    y = test_ds.iloc[:,4:5].values

    #Combining training and testing set and using the number of prediction days before the test set
    total_ds = pd.concat((train_ds['Close'],test_ds['Close']),axis=0) 
    testing_set = total_ds[ len(total_ds) -len(test_ds) - prediction_days: ].values
    testing_set = testing_set.reshape(-1,1)

    #Scaling
    testing_set = scaler.transform(testing_set)

    #Create testing data structure
    x_test=[]
    
    for i in range(prediction_days,len(testing_set)):
        x_test.append(testing_set[i-prediction_days:i,0])
    
    #Convert to numpy arrays
    x_test=np.array(x_test)

    #Reshaping: Adding 3rd dimension
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #Testing Prediction
    y_test = model.predict(x_test)

    #Getting original prices back from scaled values
    y_test = scaler.inverse_transform(y_test)
    fig = plt.figure(figsize=(7.2,4.8),dpi=65)
    plt.plot(y,label='Actual Price')  
    plt.plot(y_test,label='Predicted Price')

    plt.legend(loc=4)
    RNN_filename = ('RNNplots/' + str(ticker) + ' ' + str(date.today())  +' RNN model.png')
    plt.savefig(RNN_filename)
    plt.close(fig)

    rmse = math.sqrt(mean_squared_error(y, y_test))

    #Forecasting Prediction
    y_pred = model.predict(X_forecast)

    #Getting original prices back from scaled values
    y_pred = scaler.inverse_transform(y_pred)

    nextday_price = y_pred[0,0]

    print("Tomorrow's ",ticker," Closing Price Prediction by LSTM: ", nextday_price)
    print("LSTM RMSE:", rmse)
    return nextday_price, rmse, RNN_filename

#Twitter sentiment analysis
def analyze_tweet_sentiment(ticker):
    """
    Function which will search through twitter for the requested ticker and
    analyze the overall sentiment if positive or negative.
    It will return the overall sentiment score, the overall verdict, number of positive tweets,
    number of negative tweets and number of neutral tweets, a list of tweets and its polarities,
    the file path for the sentiment analysis pie chart image
    """
    #Find the company name associated to the ticker via yfinance
    name = yf.Ticker(ticker).info['shortName']
    
    #Accessing and authenticating Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    user = tweepy.API(auth, wait_on_rate_limit = True)
    
    #Number of tweets to analyze
    n_tweets = 300
    
    #Search twitter
    tweets = tweepy.Cursor(user.search_tweets, q=name,
                           tweet_mode='extended', lang='en').items(n_tweets)

    tweet_list = [] #List of tweets
    polarity_list =[] #List of polarities of the tweets
    overall_polarity = 0

    #Count positive and negative tweets
    positive_tweets = 0
    negative_tweets = 0
    
    for tw in tweets:

        #Convert to Textblob format for assigning polarity
        tweet = tw.full_text
       
        #Clean
        tweet = preprocess.clean(tweet)
        tweet = re.sub('&amp;','&',tweet) #replace &amp by '&'
        tweet = re.sub(':','',tweet)#Remove :
        tweet = tweet.encode('ascii', 'ignore').decode('ascii') #Remove nonascii characters

        tweet_list.append(tweet)

        blob = TextBlob(tweet)
        tweet_polarity = 0 #Polarity for each tweet
        
        #Analyze each sentence in the tweet
        for sentence in blob.sentences:
            tweet_polarity += sentence.sentiment.polarity
            
            #Increment the count whether it is positive or negative
            if tweet_polarity > 0:
                positive_tweets += 1
                
            if tweet_polarity < 0:
                negative_tweets += 1

            overall_polarity += sentence.sentiment.polarity
        
        polarity_list.append(tweet_polarity)

    if len(tweet_list) != 0:
        overall_polarity = overall_polarity / len(tweet_list)
    else:
        overall_polarity = overall_polarity
        
    neutral_tweets = n_tweets - (positive_tweets + negative_tweets)
    
    if neutral_tweets < 0:
        negative_tweets = negative_tweets + neutral_tweets
        

    print("Positive Tweets :", positive_tweets, "Negative Tweets :", negative_tweets,
          "Neutral Tweets :", neutral_tweets)

    labels=['Positive','Negative','Neutral']
    colors = ['tab:green', 'tab:red' , 'tab:orange']
    sizes = [positive_tweets, negative_tweets, neutral_tweets]
    explode = (0, 0, 0)
    fig = plt.figure(figsize=(7.2,4.8),dpi=65)
    fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    
    SA_filename = 'SApiecharts/'+ str(ticker) +' '+ str(date.today()) +' Twitter Sentiment Analysis.png'
    plt.savefig(SA_filename)
    plt.close(fig)

    #plt.show()
    if overall_polarity > 0:
        polarity_verdict = 'Overall Positive'
    else:
        polarity_verdict = 'Overall Negative'
        
    return overall_polarity, polarity_verdict, positive_tweets, negative_tweets, neutral_tweets, tweet_list, polarity_list ,SA_filename

def recommend_action(polarity, info_ticker, price_nextday):
    if info_ticker.iloc[-1]['Close'] < price_nextday:
        if polarity > 0:
            decision = 'Good sentiment and rising. Seems like a good idea to buy.'
        elif polarity <= 0:
            decision = "Bad sentiment and rising. Might wait before buying or sell some existing stock."
    elif info_ticker.iloc[-1]['Close'] > price_nextday:
        if polarity > 0:
            decision= 'Good sentiment and falling. Might wait before buying.'
        elif polarity <= 0:
            decision= 'Bad sentiment and falling. Seems like a good idea to sell.'  
    
    return decision

#Main execution

#Title
st.title("Stock Prediction with Neural Network and Twitter NLP sentiment analysis")


#Search ticker
ticker = st.text_input('Type in the selected ticker ', '')
search_button = st.button('Search')

if search_button:
    
    ticker = ticker.upper()
    
    #Fetching and saving the ticker info into CSV
    data_load_state = st.text("Loading data...")
    csv_path, ticker_name = get_quote(ticker)
    df = pd.read_csv(csv_path)
    data_load_state.text("Loading data...Done!")
    
    #Read and diplay the data
    st.subheader("Today's " + ticker_name +' ('+ ticker + ") information for " + str(date.today()))
    st.table(df.tail(1))
    df = df.dropna()
    
    #Plot and display the ticker
    def plot_ticker_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name = 'Close Price'))
        fig.layout.update(title_text=ticker + " Time Series", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_ticker_data()
    
    #Predicting the stock price
    st.subheader(ticker + " Model Price Prediction")
    predict_state = st.text("Predicting...")
    price_nextday, rmse, RNN_filename = predict_price(df, ticker)
    predict_state.text("Predicting...Done!")
    
    image_RNN = Image.open(RNN_filename)
    st.image(image_RNN, caption = ticker + ' Past 100 days Real vs Predicted Price') #Display Real vs Predicted plot
    st.write("Predicted price at the closing of the next stock day: " + str(price_nextday))
    st.write("The model RMSE is at: " + str(rmse))
    
    #Twitter Sentiment Analysis
    st.subheader(ticker_name + " Twitter Sentiment Analysis")
    twitter_search_state = st.text("Searching through Twitter...")
    polarity, polarity_verdict, positive, negative, neutral, tweet_list, polarity_list, SA_filename = analyze_tweet_sentiment(ticker)
    twitter_search_state.text("Searching through Twitter...Done!")
    
    image_SA = Image.open(SA_filename)
    st.image(image_SA, caption = 'Twitter Sentiment Pie Chart for ' + ticker_name) #Display Sentiment Analysis Pie Chart
    
    total = positive + negative + neutral
    st.write("Number of positive tweets: " + str(positive) + ' ( '+ str(round((positive/total)*100,2))  +'% )')
    st.write("Number of neutral tweets: " + str(neutral) + ' ( '+ str(round((neutral/total)*100,2))  +'% )')
    st.write("Number of negative tweets: " + str(negative) + ' ( '+ str(round((negative/total)*100,2))  +'% )')
    
    st.write("A few examples of tweets:")
    tweet_df = pd.DataFrame(list(zip(tweet_list, polarity_list)), columns = ['Tweet', 'Polarity'])
    st.write(tweet_df.head(10))
    
    st.write(ticker + ' Overall Polarity: ' + str(polarity) + " = " + polarity_verdict)
    
    st.subheader("Reommendation for " + ticker)
    recommend = recommend_action(polarity, df, price_nextday)
    st.write(recommend)





