from django.shortcuts import render

# Create your views here.
# from django.shortcuts import render
from django.http import HttpResponse, Http404
from django.core.serializers import serialize
from .models import StockData


import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pickle
from joblib import Parallel, delayed
import joblib

import os


# Create your views here.
def index(request):
    tickers = ['NVDA','AMD','INTC']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    StockData.objects.all().delete()
    
    for ticker in tickers:
        data = fetch_yahoo_data(ticker, start_date, end_date)
        save_to_django_model(data, ticker)
    #print(StockData.objects.all())
        
    return render(request, "index.html")


def fetch_yahoo_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def save_to_django_model(data, symbol):
    i = 2
    for index, row in data.iterrows():
        StockData.objects.create(
            symbol=symbol,
            date=index,
            open_price=row['Open'],
            high_price=row['High'],
            low_price=row['Low'],
            close_price=row['Close'],
            volume=row['Volume']
        )
        if(i != 0):
            print("Symbol - ", symbol)
            print("Date - ", index)
            print("Open_price - ", row['Close'])
            i -=1 
        
        
def service(request):
    #return render(request, "page2.html")
    
    data = StockData.objects.all()
    serialized_data = serialize('json', data)
    print("######### MY SERIALIZED DATA ###############")
    return render(request, 'service.html', {'data': serialized_data})


def trend_prediction(request):
    
    # AMD PRICE PRED BEGIN
    stock_symbol = 'AMD'
    
    data = yf.download(tickers=stock_symbol,period='100d',interval='1d')

    close = data[['Close']]
    ds = close.values


    #Inverse transform to get actual value
    file_name = 'zeeapp/normalizer_amd.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    
    
    scaler = joblib.load(absolute_path)

    ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test data sizes
    test_size = len(ds_scaled)

    #Assigning test data
    ds_test = ds_scaled

    #creating dataset in time series for LSTM model
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for prediction
    time_stamp = 100
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Predict using model pkl file
    file_name = 'zeeapp/lstm_model_amd.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    lstm_model = joblib.load(absolute_path)




    #Getting the last 100 days records

    fut_inp = ds_test.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    print(lst_output)

    print(scaler.inverse_transform(lst_output))
    
    ans = scaler.inverse_transform(lst_output)
    
    res_amd = ans.tolist()
    print(type(ans))
    print(type(res_amd))
    # AMD PRED ENDS
    
    #NVDA PRED BEGINS
    stock_symbol = 'NVDA'
    
    data = yf.download(tickers=stock_symbol,period='100d',interval='1d')

    close = data[['Close']]
    ds = close.values


    #Inverse transform to get actual value
    file_name = 'zeeapp/normalizer_nvda.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    
    
    scaler = joblib.load(absolute_path)

    ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test data sizes
    test_size = len(ds_scaled)

    #Assigning test data
    ds_test = ds_scaled

    #creating dataset in time series for LSTM model
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for prediction
    time_stamp = 100
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Predict using model pkl file
    file_name = 'zeeapp/lstm_model_nvda.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    lstm_model = joblib.load(absolute_path)




    #Getting the last 100 days records

    fut_inp = ds_test.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    print(lst_output)

    print(scaler.inverse_transform(lst_output))
    
    ans = scaler.inverse_transform(lst_output)
    
    res_nvda = ans.tolist()
    print(type(ans))
    print(type(res_nvda))
    
    
    #NVDA PRED ENDS
    
    
    #INTL PRED BEGINS
    stock_symbol = 'INTC'
    
    data = yf.download(tickers=stock_symbol,period='100d',interval='1d')

    close = data[['Close']]
    ds = close.values


    #Inverse transform to get actual value
    file_name = 'zeeapp/normalizer_intc.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    
    scaler = joblib.load(absolute_path)

    ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test data sizes
    test_size = len(ds_scaled)

    #Assigning test data
    ds_test = ds_scaled

    #creating dataset in time series for LSTM model
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for prediction
    time_stamp = 100
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Predict using model pkl file
    file_name = 'zeeapp/lstm_model_intc.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    lstm_model = joblib.load(absolute_path)




    #Getting the last 100 days records

    fut_inp = ds_test.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    print(lst_output)

    print(scaler.inverse_transform(lst_output))
    
    ans = scaler.inverse_transform(lst_output)
    
    res_intl = ans.tolist()
    print(type(ans))
    print(type(res_intl))
    
    #INTL PRED ENDS
    
    
    context = {
        'res_nvda' : res_nvda,
        'res_amd' : res_amd,
        'res_intl' : res_intl
    }
    
    
    
    return render(request, 'trend_prediction.html', context)


def prediction(request):
    
    # AMD PRICE PRED BEGIN
    stock_symbol = 'AMD'
    
    data = yf.download(tickers=stock_symbol,period='100d',interval='1d')

    close = data[['Close']]
    ds = close.values


    #Inverse transform to get actual value
    file_name = 'zeeapp/normalizer_amd.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    
    scaler = joblib.load(absolute_path)

    ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test data sizes
    test_size = len(ds_scaled)

    #Assigning test data
    ds_test = ds_scaled

    #creating dataset in time series for LSTM model
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for prediction
    time_stamp = 100
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Predict using model pkl file
    file_name = 'zeeapp/lstm_model_amd.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    lstm_model = joblib.load(absolute_path)




    #Getting the last 100 days records

    fut_inp = ds_test.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    print(lst_output)

    print(scaler.inverse_transform(lst_output))
    
    ans = scaler.inverse_transform(lst_output)
    
    res_amd = ans.tolist()
    print(type(ans))
    print(type(res_amd))
    # AMD PRED ENDS
    
    #NVDA PRED BEGINS
    stock_symbol = 'NVDA'
    
    data = yf.download(tickers=stock_symbol,period='100d',interval='1d')

    close = data[['Close']]
    ds = close.values


    #Inverse transform to get actual value
    file_name = 'zeeapp/normalizer_nvda.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    
    scaler = joblib.load(absolute_path)

    ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test data sizes
    test_size = len(ds_scaled)

    #Assigning test data
    ds_test = ds_scaled

    #creating dataset in time series for LSTM model
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for prediction
    time_stamp = 100
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Predict using model pkl file
    file_name = 'zeeapp/lstm_model_nvda.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    lstm_model = joblib.load(absolute_path)




    #Getting the last 100 days records

    fut_inp = ds_test.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    print(lst_output)

    print(scaler.inverse_transform(lst_output))
    
    ans = scaler.inverse_transform(lst_output)
    
    res_nvda = ans.tolist()
    print(type(ans))
    print(type(res_nvda))
    
    
    #NVDA PRED ENDS
    
    
    #INTL PRED BEGINS
    stock_symbol = 'INTC'
    
    data = yf.download(tickers=stock_symbol,period='100d',interval='1d')

    close = data[['Close']]
    ds = close.values


    #Inverse transform to get actual value
    file_name = 'zeeapp/normalizer_intc.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    
    scaler = joblib.load(absolute_path)

    ds_scaled = scaler.fit_transform(np.array(ds).reshape(-1,1))

    #Defining test data sizes
    test_size = len(ds_scaled)

    #Assigning test data
    ds_test = ds_scaled

    #creating dataset in time series for LSTM model
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
            a = dataset[i:(i+step), 0]
            Xtrain.append(a)
            Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)

    #Taking 100 days price as one record for prediction
    time_stamp = 100
    X_test, y_test = create_ds(ds_test,time_stamp)

    #Predict using model pkl file
    file_name = 'zeeapp/lstm_model_intc.pkl'
    absolute_path = os.path.abspath(os.path.join(os.getcwd(), file_name))
    lstm_model = joblib.load(absolute_path)




    #Getting the last 100 days records

    fut_inp = ds_test.reshape(1,-1)
    tmp_inp = list(fut_inp)
    tmp_inp = tmp_inp[0].tolist()

    #Predicting next 30 days price suing the current data
    #It will predict in sliding window manner (algorithm) with stride 1
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(tmp_inp)>100):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = lstm_model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1


    print(lst_output)

    print(scaler.inverse_transform(lst_output))
    
    ans = scaler.inverse_transform(lst_output)
    
    res_intl = ans.tolist()
    print(type(ans))
    print(type(res_intl))
    
    #INTL PRED ENDS
    
    arr = [1,2,3,4,5]
    context = {
        'val1' : res_nvda[0],
        'val2' : res_nvda[1],
        'val3' : res_nvda[2],
        'val4' : res_nvda[3],
        'val5' : res_nvda[4],
        'val6' : res_amd[0],
        'val7' : res_amd[1],
        'val8' : res_amd[2],
        'val9' : res_amd[3],
        'val10' : res_amd[4],
        'val11' : res_intl[0],
        'val12' : res_intl[1],
        'val13' : res_intl[2],
        'val14' : res_intl[3],
        'val15' : res_intl[4],
        
    }
    
    return render(request, 'prediction.html', context)
