import numpy as np
from numpy.random import seed
seed(1)
from numpy import array

import pandas as pd
import datetime
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
tf.random.set_seed(2)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Bidirectional

import math
import itertools

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#load datasets
datafile = "/content/sample_data/data.npz"
data = np.load(datafile)
train = data['train'] #2014-2018
validation = data['validation'] #2019
test = data['test'] #2020

                #create index and concatenate all sets into one
dti = pd.date_range(start='2014-01-01 00:00:00', periods=len(np.concatenate((data['train'], data['validation'], data['test']))), freq='H')
#dti
ts_complete = pd.Series(np.concatenate((data['train'], data['validation'], data['test'])), index=dti)
#ts_complete.describe(include="all")
#ts_complete.plot()
df= ts_complete.to_frame(name="Power")

    #Scale data between 0-1
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(df)


    #train test split at 70% vs. 30%
train_size = int(len(ts_scaled) * 0.7)
test_size = len(ts_scaled) - train_size
train, test = ts_scaled[0:train_size,:], ts_scaled[train_size:len(ts_scaled),:]
plt.figure(figsize=(15,6))
plt.plot(train)

plt.figure(figsize=(15,6))
plt.plot(test)

#create the data slicing by defining the number of data points to be taken in and out
def split_data(dataset, split_no, steps_out):
    dataX, dataY = [], []
    for i in range(len(dataset)-split_no-steps_out):
        a = i+split_no
        b = a+steps_out
        dataX.append(dataset[i:a,0])
        dataY.append(dataset[a:b,0])
    dataX, dataY= np.array(dataX), np.array(dataY)
    return dataX, dataY

#performs the split of the sets and reshapes to be fed into the NN.
def test_train_reshape(train_set,test_set, split_no, steps_out):
    global train_X, train_Y
    train_X, train_Y = split_data(train, split_no, steps_out)
    global test_X, test_Y
    test_X, test_Y= split_data(test, split_no, steps_out)
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1],1))
    test_X = np.reshape(test_X, (test_X.shape[0],test_X.shape[1],1))

#compute the Mean Absolute Percentage Error such that the
#computation takes each step, resulting in i MAPE outputs
def multi_step_MAPE(y_true, y_pred):
  global mape1
  mape1=[]
  for i in range(y_true.shape[1]):
    true1, pred1= y_true[:,i], y_pred[:,i]
    compute_mape=(np.mean(np.abs((true1 - pred1) /true1)) * 100)
    mape1.append(np.round(compute_mape,2))

#RMSE by time step 
def multi_step_RMSE(y_true, y_pred):
  global rmse_1
  rmse_1=[]
  for i in range(y_true.shape[1]):
      true1, pred1= y_true[:,i], y_pred[:,i]
      compute_rmse=(math.sqrt(mean_squared_error(true1,pred1)))
      rmse_1.append(np.round(compute_rmse,2))

#model to be ran given the different time "slicings"
def run_lstm(train_X, train_Y, test_X, test_Y, epochs_no, batch_no, neurons):
    #prepare model
    model0 = Sequential()
    #input shape refers to the time steps to learn from and the columns of the data
    #therefore, for univariate data (steps_in, 1)
    model0.add(Bidirectional(LSTM(neurons, activation='relu',return_sequences=True, input_shape=(train_X.shape[1],1))))
    model0.add(Flatten())
    #output layers takes the number of output steps defined through the data_split function
    model0.add(Dense(steps_out))
    model0.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    #fit model
    model0.fit(train_X, train_Y, epochs=epochs_no, batch_size=batch_no, verbose=2) 

    global trainPredict, testPredict, train_YS, test_YS
    trainPredict = model0.predict(train_X)
    testPredict = model0.predict(test_X)
       
    #scale back to original scale
    trainPredict = scaler.inverse_transform(trainPredict)
    train_YS=scaler.inverse_transform(train_Y)
    testPredict = scaler.inverse_transform(testPredict)
    test_YS = scaler.inverse_transform(test_Y)
    
    #Obtain the RMSE and MAPE to be recorded in a DF
    global rmse_report
    rmse_report=pd.DataFrame()
    #init. arrays to append the corresponding RMSE and MAPE
    trainScore,testScore, MAPE= [],[],[]

    #Train RMSE scores
    trainScore.append(multi_step_RMSE(train_YS, trainPredict))
    #create column names according to the number of time steps used
    col_names2=["TrainRMSE " +str(i+1) for i in range(train_YS.shape[1])]
    #append col names and TrainRMSE score to report
    for i in range(train_YS.shape[1]):
      rmse_report[col_names2[i]]=[rmse_1[i]]

    #Test RMSE scores
    testScore.append(multi_step_RMSE(test_YS, testPredict))
    #create column names according to the number of time steps used
    col_names3=["TestRMSE " +str(i+1) for i in range(test_YS.shape[1])]
    #append col names and TrainRMSE score to report
    for i in range(test_YS.shape[1]):
      rmse_report[col_names3[i]]=[rmse_1[i]]

    #compute the MAPE for each time step
    MAPE.append(multi_step_MAPE(test_YS,testPredict))
    #create column names according to the number of time steps used
    col_names=["MAPE "+str(i+1) for i in range(test_YS.shape[1])]
    #append col names and MAPE to report
    for i in range(test_YS.shape[1]):
      rmse_report[col_names[i]]=[mape1[i]]

######## j Epochs, k Neurons, i batches ###########

        #Weekly split 24x7 - 168
#define batches, epochs, and neurons to be implemented
K=46 #batches
L=200 #epochs
M=32 #neurons 

#learning sequence by week (steps in)
steps_in=168
#prediction seq (steps out)
steps_out=3

#run data split and necessary reshapes 
test_train_reshape(train,test,steps_in, steps_out)

#crate empty df to record metrics and configuration
rmse0=pd.DataFrame()

#init. timer
start_time = datetime.now()

#run model
j_epochs,i_batch, k_neurons= list(),list(),list()
run_lstm(train_X, train_Y, test_X, test_Y,
             epochs_no=L, 
             batch_no=K, 
             neurons=M)

rmse0=rmse0.append([rmse_report])
i_batch.append(K)
j_epochs.append(L)
k_neurons.append(M)
         
rmse0['Neurons'],rmse0['Epochs'], rmse0['Batches']=k_neurons,j_epochs, i_batch
rmse0=rmse0.sort_values(by=['Neurons','Epochs', 'Batches'], ascending=[True,True,True])
rmse0=rmse0.reset_index(drop=True)
rmse0

end_time = datetime.now()
print('Implementation Time: {}'.format(end_time - start_time))

#Save DF with results as CSV file
rmse0.to_csv(r'Weekly_Bi_3stepLSTM_Results.csv',index=False, header=True)