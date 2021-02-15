import numpy as np
from numpy.random import seed
seed(1)
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
plt.show()
plt.figure(figsize=(15,6))
plt.plot(test)
plt.show()

#create the data slicing by defining the number of data points to be taken
def split_data(dataset, steps):
    dataX, dataY = [], []
    for i in range(len(dataset)-steps-1):
        a = dataset[i:(i+steps), 0]
        dataX.append(a)
        dataY.append(dataset[i + steps, 0])
    return np.array(dataX), np.array(dataY)

#train/test for LSTM given the time window/split_no to be used (168,720,2160)
#performs the split of the sets and reshapes to be fed into the NN.
def test_train_reshape(train_set,test_set, split_no):
    global train_X, train_Y
    train_X, train_Y = split_data(train, split_no)
    global test_X, test_Y
    test_X, test_Y= split_data(test, split_no)
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    #return train_X, train_Y, test_X, test_Y

#compute de Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#model to be ran given the different time "slicings"
def run_lstm(train_X, train_Y, test_X, test_Y, epochs_no, batch_no, neurons_layer1, neurons_layer2):
    #prepare model
    model1 = Sequential()
    model1.add(LSTM(neurons_layer1, activation='relu',return_sequences=True, input_shape=(1,train_X.shape[2])))
    model1.add(LSTM(neurons_layer2, activation='relu'))
    #model1.add(Flatten()) - incompatible with stacked, no need for flatten output before dense layer
    model1.add(Dense(1))
    model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    
    #init. lists to append the corresponding RMSE and MAPE
    trainScore,testScore, MAPE= list(),list(),list()
    #fit model
    model1.fit(train_X, train_Y, epochs=epochs_no, batch_size=batch_no, verbose=2) 

    global trainPredict, testPredict, train_YS, test_YS
    trainPredict = model1.predict(train_X)
    testPredict = model1.predict(test_X)
       
    #scale back to original scale
    trainPredict = scaler.inverse_transform(trainPredict)
    train_YS = scaler.inverse_transform([train_Y])
    testPredict = scaler.inverse_transform(testPredict)
    test_YS = scaler.inverse_transform([test_Y])
    
    #Obtain the RMSE and MAPE to be recorded in a DF
    train_YS=train_YS.reshape(-1,1)
    test_YS=test_YS.reshape(-1,1)
    trainScore.append(math.sqrt(mean_squared_error(train_YS, trainPredict)))
    testScore.append(math.sqrt(mean_squared_error(test_YS, testPredict)))
    MAPE.append(mean_absolute_percentage_error(test_YS,testPredict))
    global rmse_report
    rmse_report=pd.DataFrame()
    rmse_report['Train'], rmse_report['Test'], rmse_report['MAPE']= trainScore, testScore, MAPE

######## j Epochs, k Neurons, i batches ###########
        # Weekly split 24x7 - 168

#define batches, epochs, and neurons to be implemented
K=[12,24,46]
L=[100,200,300]
M=[32,64]

#Split and reshape data
test_train_reshape(train,test,168)

#run model
rmse0=pd.DataFrame()
j_epochs,i_batch, k_neurons1, k_neurons2= list(), list(), list(), list()
start_time = datetime.now()
for i,j,k,l in itertools.product(K, L, M, M):
    run_lstm(train_X, train_Y, test_X, test_Y,
             batch_no=i,
             epochs_no=j,  
             neurons_layer1=k,
             neurons_layer2=l)
    rmse0=rmse0.append([rmse_report])
    j_epochs.append(j)
    i_batch.append(i)
    k_neurons1.append(k)
    k_neurons2.append(l)
        
rmse0['Neurons Layer 1'],rmse0['Neurons Layer 2'],rmse0['Epochs'], rmse0['Batches']=k_neurons1,k_neurons2,j_epochs, i_batch
rmse0=rmse0.sort_values(by=['Neurons Layer 1','Epochs', 'Batches'], ascending=[True,True,True])
rmse0=rmse0.round({'Train':2,'Test':2,'MAPE':2})
rmse0=rmse0.reset_index(drop=True)
rmse0

end_time = datetime.now()
print('Implementation Time: {}'.format(end_time - start_time))

#Save DF with results as CSV file
rmse0.to_csv(r'Weekly_StkLSTM_Results.csv',
            index=False, header=True)

#record df index where the lowest MAPE is recorded
#rmse0=pd.read_excel('/content/sample_data/Weekly_StkLSTM_Results.xlsx')
id=rmse0[['MAPE']].idxmin() 
#get row given index
min_df=rmse0.iloc[id]
#given the configuration which yielded the lowest MAPE run LSTM model with said
#parameters
test_train_reshape(train,test,168)
start_time=datetime.now()
run_lstm(train_X, train_Y, test_X, test_Y,
             batch_no=min_df['Batches'].values[0],
             epochs_no=min_df['Epochs'].values[0],
             neurons_layer1=min_df['Neurons Layer 1'].values[0],
             neurons_layer2=min_df['Neurons Layer 2'].values[0]
             )


end_time = datetime.now()
print('Implementation Time: {}'.format(end_time - start_time))

# shift test predictions for plotting
testPredictPlot = np.empty_like(ts_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(168*2)+1:len(ts_scaled)-1, :] = testPredict
trainPredictPlot = np.empty_like(ts_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[168:len(trainPredict)+168, :] = trainPredict

trainsc=np.empty_like(ts_scaled)
trainsc[:, :] = np.nan
trainsc[168:len(ts_scaled)+168, :]=ts_scaled[168:len(ts_complete)+168, :]

testsc= np.empty_like(ts_scaled)
testsc[:, :] = np.nan
testsc[len(trainPredict)+(168*2)+1:len(ts_scaled)-1, :]=ts_scaled[len(trainPredict)+(168*2)+1:len(ts_scaled)-1, :]

# plot baseline and predictions
plt.figure(figsize=(25,8))
plt.plot(scaler.inverse_transform(trainsc), color='tab:blue')
#plt.legend("True Values")
plt.plot(trainPredictPlot,color="orange")
plt.plot(testPredictPlot,color='yellowgreen')
plt.legend(["True Values","Training Prediction","Test Prediction"], loc='upper left', frameon=False)
plt.ylabel('MWh')
plt.show()

#Plot time series of observed values versus predicted values through the 
#training and testing sets
# shift test predictions for plotting

trainPredictPlot = np.empty_like(ts_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[168:len(trainPredict)+168, :] = trainPredict
trainsc=np.empty_like(ts_scaled)
trainsc[:, :] = np.nan
trainsc[168:len(trainPredict)+168, :]=train[168:len(trainPredict)+168, :]

# plot baseline and predictions
import seaborn as sns
sns.set(rc={'figure.figsize':(15,5)},font_scale = 2)
fig, ax=plt.subplots()

ax.plot(scaler.inverse_transform(trainsc[:20000]))
ax.plot(trainPredictPlot[:20000], color="orange")
ax.legend(["True Values","Training Prediction Sample"],loc='upper left', frameon=False)
ax.set_ylabel('MWh')


# shift test predictions for plotting
testPredictPlot = np.empty_like(ts_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(168*2)+1:len(ts_scaled)-1, :] = testPredict
trainPredictPlot = np.empty_like(ts_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[168:len(trainPredict)+168, :] = trainPredict

testsc= np.empty_like(ts_scaled)
testsc[:, :] = np.nan
testsc[len(trainPredict)+(168*2)+1:len(ts_scaled)-1, :]=ts_scaled[len(trainPredict)+(168*2)+1:len(ts_scaled)-1, :]

# plot baseline and predictions
import seaborn as sns
sns.set(rc={'figure.figsize':(15,5)},font_scale = 2)
fig, ax=plt.subplots()

ax.plot(scaler.inverse_transform(testsc), color='tab:blue')
ax.plot(testPredictPlot, color='yellowgreen')
ax.legend(["True Values","Test Prediction"],loc='upper left', frameon=False)
ax.set_ylabel('MWh')