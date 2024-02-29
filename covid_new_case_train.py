import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.layers import Bidirectional, Embedding
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
#from Covid19_newcase_analysis_module import ExploratoryDataAnalysis
#from Covid19_newcase_analysis_module import ModelCreation
#from Covid19_newcase_analysis_module import ModelEvaluation
FILE_PATH = os.path.join(os.getcwd(), 'Data', 'Covid-19_case\cases_malaysia_covid_FULL.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'Saved_path', 'model.h5')
LOG_PATH = os.path.join(os.getcwd(), 'Log')

# Constant
window_size = 30 # how many day of prediction

#%% EDA
# Load data
df = pd.read_csv(FILE_PATH)

# Data inspection
df.isna().sum()  # Check NA value
df.duplicated().sum()  # Check for duplicates

#(A) Convert cases_new column from object to int64
df['cases_new'] = df['cases_new'].replace([' ','?'],'0') # replace ' ' and '?' to '0'
df['cases_new'] = df['cases_new'].fillna(0) # fill the null value with '0'
df['cases_new'] = df[('cases_new')].astype('int64') # change 

#(B) Fill the remaining NaN data with 0
fill_values = {'cluster_import':0,'cluster_religious':0,'cluster_community':0,'cluster_highRisk':0,'cluster_education':0,'cluster_detentionCentre':0,'cluster_workplace':0,}
df.fillna(fill_values, inplace=True)

# Train, validation, test split
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[:680]
test_df = df[681:]

class ExploratoryDataAnalysis():
    
    def knn_imputer(self, data ,n_neighbors=2):     
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
        plt.figure()
        plt.plot(data2)
        plt.plot(data1)
        plt.legend(['Predicted', 'Actual'])
        plt.show()
        
        mean_absolute_error(data1, data2)

        print('\n Mean absolute percentage error:', 
              mean_absolute_error(data1, data2)/sum(abs(data1))*100)

eda = ExploratoryDataAnalysis()
train_df_clean = eda.knn_imputer(train_df,n_neighbors=24)
test_df_clean = eda.knn_imputer(test_df,n_neighbors=24)

# Step 5) Feature selection
# Step 6) Data preprocessing
# In this analysis, we only choose 'cases_new' with index 0
scaled_train_df = eda.mm_scaler(train_df_clean, index_column=0)
scaled_test_df = eda.mm_scaler(test_df_clean, index_column=0)

# Testing dataset with windows 30 day
window_size = 30

# train data
#eda.train_process_window(train_df, scaled_train_df)
X_train=[]
Y_train=[]

for i in range(window_size, len(train_df)):
    X_train.append(scaled_train_df[i-window_size:i,0])
    Y_train.append(scaled_train_df[i,0])
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# test data
temp = np.concatenate((scaled_train_df, scaled_test_df))
length_window = window_size+len(scaled_test_df)
temp = temp[-length_window:] 

#eda.train_process_window(temp, window_size=30)
X_test=[]
Y_test=[]

for i in range(window_size, len(temp)):
    X_test.append(temp[i-window_size:i,0])
    Y_test.append(temp[i,0])
    
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# expend dimension
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
  
#%% Model creation
mc = ModelCreation()
model = mc.lstm_layer(X_train, nodes=64, dropout=0.2, output=1)

model.compile(optimizer='adam', loss='mse', metrics='mse')

# callback
log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir)
early_stopping_callback = EarlyStopping(monitor='loss', patience=15)

hist = model.fit(X_train, Y_train, epochs=200, batch_size=200,
                 validation_data=(X_test,Y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])

print(hist.history.keys())

#%% model deployment
predicted = [] 

for test in X_test:
    predicted.append(model.predict(np.expand_dims(test,axis=0)))

predicted = np.array(predicted)

y_true = Y_test
y_pred = predicted.reshape(len(predicted),1)

#%% Model Analysis
me = ModelEvaluation()
me.model_report(y_true, y_pred)

#%% Save model
model.save(MODEL_SAVE_PATH)