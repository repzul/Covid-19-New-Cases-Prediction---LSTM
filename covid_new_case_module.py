# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:39:34 2022

@author: snaff
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.layers import Bidirectional, Embedding
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler

class ExploratoryDataAnalysis():
    
    def knn_imputer(self, data ,n_neighbors=2):
        '''
        This function do imputation process to fill in NaN value in dataframe

        Parameters
        ----------
        data : Array
            Array inside the dataframe
            
        n-neighbors : Int
            No of column it will compare too. Default is 2

        Returns
        -------
        data : Array
            Return array in dataframe

        '''
               
        imputer = KNNImputer(n_neighbors=n_neighbors)
        temp = data.drop(labels=['date'], axis=1) 
        temp_date = data['date']
        df_imputed = imputer.fit_transform(temp) #result in float, turn into array
        # Convert it to dataframe with int datatype
        df_imputed = pd.DataFrame(df_imputed.astype('int'))
        
        # Combine back all the columns
        train_df_clean = pd.concat((temp_date,df_imputed),axis=1)
        
        return train_df_clean
        
    def mm_scaler(self, data, index_column):
        '''
        This function will do Min Max Scaler on the data, scale it and
        expand the dimension

        Parameters
        ----------
        data : Array
            Cleaned training data

        Returns
        -------
        data: Array
            Data with expended dimension

        '''
        scaler = MinMaxScaler()
        data = data[index_column].values
        scaled_data = scaler.fit_transform(np.expand_dims(data, axis=-1))
        
        return scaled_data
    
    def train_process_window(self, data1, data2, window_size=30):
        
        X_train=[]
        Y_train=[]
        
        for i in range(window_size, len(data1)):
            X_train.append(data2[i-window_size:i,0])
            Y_train.append(data2[i,0])
            
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        return X_train, Y_train
    
    def test_process_window(self, data, window_size=30):
        
        X_test=[]
        Y_test=[]
        
        for i in range(window_size, len(data)):
            X_test.append(data[i-window_size:i,0])
            Y_test.append(data[i,0])
            
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        
        return X_test, Y_test

    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self, data, nodes=64, dropout=0.3, output=1):
        '''
        This function is to creates a LSTM model with 2 hidden layers. 
        Last layer of the model comrises of tanh activation function
     
        Parameters
        ----------
        nodes : Int, optional
            DESCRIPTION. The default is 64
        dropout : Float, optional
            DESCRIPTION. The default is 0.3
     
        Returns
        -------
        Model: Created Model

        '''

        model = Sequential()
        model.add(LSTM(nodes, activation='tanh',return_sequences=(True),
                        input_shape=(data.shape[1],1)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(LSTM(nodes)) 
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(output))
        model.summary()
            
        return model
        
class ModelEvaluation():
    def model_report(self, data1, data2):
        '''
        This function is to evaluate the model created. 
        1. Plot the graph
        2. Print the mean_absolute_error

        Parameters
        ----------
        y_true : Array
            True value in array
        y_pred : Array
            Prediction value in array

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(data2)
        plt.plot(data1)
        plt.legend(['Predicted', 'Actual'])
        plt.show()
        
        mean_absolute_error(data1, data2)

        print('\n Mean absolute percentage error:', 
              mean_absolute_error(data1, data2)/sum(abs(data1))*100)
        