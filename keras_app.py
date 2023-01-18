import streamlit as st 
#st.markdown(""" This is a Streamlit App """)
import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px

#TO DO: get the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()










mT = x_train.mean()
sT = x_train.std()
x_test_anomalies =[np.random.normal(mT,sT,784).reshape(28,28)]

for x in range(0,500):
  new_x=np.random.normal(mT,sT,784).reshape(28,28) +x_test[x+1].reshape(28,28)+x_test[x+3].reshape(28,28)
  x_test_anomalies=np.append(x_test_anomalies, [new_x], axis=0) #+x_test[x+3].reshape(28,28)

for i in range(0,25):
  j=2*i
  x_test[j]=x_test_anomalies[i]


# 

#TO DO: reshaping:
# 
# The new shape should be a 2d array having 60000 records described as 1d arrays
# 

feature_vector_lenght= x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0],feature_vector_lenght)
x_test = x_test.reshape(x_test.shape[0],feature_vector_lenght)
x_test_anomalies= x_test_anomalies.reshape(x_test_anomalies.shape[0],feature_vector_lenght)





#TO DO: normalization in [0,1]
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
x_test_anomalies = x_test_anomalies.astype("float32")/255



num_classes = 10


#@title Texto de título predeterminado
#TO DO: model
def model_def(num_nodes,num_layers, act_func):
   if num_nodes == 2:
     if num_layers == 8:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(392,activation=act_func, input_shape=(784,)))
       model.add(layers.Dense(196,activation=act_func))
       model.add(layers.Dense(98,activation=act_func))
       model.add(layers.Dense(49,activation=act_func))
       #decode
       model.add(layers.Dense(98,activation=act_func))
       model.add(layers.Dense(196,activation=act_func))
       model.add(layers.Dense(392,activation=act_func))
       model.add(layers.Dense(784,activation=act_func))
     elif num_layers == 6:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(392,activation=act_func, input_shape=(784,)))
       model.add(layers.Dense(196,activation=act_func))
       model.add(layers.Dense(98,activation=act_func))
       #decode
       model.add(layers.Dense(196,activation=act_func))
       model.add(layers.Dense(392,activation=act_func))
       model.add(layers.Dense(784,activation=act_func))

     elif num_layers == 4:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(392,activation=act_func, input_shape=(784,)))
       model.add(layers.Dense(196,activation=act_func))
       #decode
       model.add(layers.Dense(392,activation=act_func))
       model.add(layers.Dense(784,activation=act_func))

     elif num_layers == 2:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(392,activation=act_func, input_shape=(784,)))
       #decode
       model.add(layers.Dense(784,activation=act_func))

   if num_nodes == 4:
   

     if num_layers == 4:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(196,activation=act_func, input_shape=(784,)))
       model.add(layers.Dense(49,activation=act_func))
       #decode
       model.add(layers.Dense(196,activation=act_func))
       model.add(layers.Dense(784,activation=act_func))

     elif num_layers == 2:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(196,activation=act_func, input_shape=(784,)))
       #decode
       model.add(layers.Dense(784,activation=act_func))

   if num_nodes == 8:
   

     if num_layers == 2:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(98,activation=act_func, input_shape=(784,)))
       #decode
       model.add(layers.Dense(784,activation=act_func))

   if num_nodes == 16:
   

     if num_layers == 2:
       model = keras.Sequential()
       #encode
       model.add(layers.Dense(49,activation=act_func, input_shape=(784,)))
       #decode
       model.add(layers.Dense(784,activation=act_func))

      




   return model


vanilla_model= model_def(2,2, 'relu')
st.title('Image prediction and anomalies detection')
st.sidebar.header('User Imput Parameters')
options_epochs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
val_epochs = st.sidebar.selectbox('Epochs', options_epochs)
option_func = ['relu', 'tanh', 'softmax', 'sigmoid']
actfun = st.sidebar.selectbox('Which activation function do you like to use?', option_func)
number_of_nodes = st.sidebar.select_slider('number of nodes', options=(2,4,8,16))
if number_of_nodes == 2:
  number_of_layers = st.sidebar.select_slider('number of layers', options=(2,4,6,8))
elif number_of_nodes == 4:
  number_of_layers = st.sidebar.select_slider('number of layers', options=(2,4))
else:
  st.sidebar.text('Do you want to continue with 2 layers?')
  continue_with_model = st.sidebar.checkbox('yes')
  back_to_selection = st.sidebar.checkbox('no')
  if continue_with_model:
    number_of_layers = 2
    st.sidebar.text('Press RUN to continue')
  elif back_to_selection:
    st.sidebar.text('Select other number of nodes')
vanilla_model= model_def(number_of_nodes,number_of_layers, actfun)
number = int(st.number_input('Introduce the element of the test set you would like to use as input', step = 1))
if st.button('RUN'):
  vanilla_model.summary()

  batch_size = 128
  epochs = val_epochs

# TO DO: Compile the model
  vanilla_model.compile(optimizer='adam', loss='mean_squared_error') #MAE
  history = vanilla_model.fit(x_train, x_train, batch_size= batch_size, epochs = epochs, validation_split= 0.1)
  st.write("Error on training set:",vanilla_model.evaluate(x_train,x_train))
  st.write("Error on test set:",vanilla_model.evaluate(x_test,x_test))
  st.write("Error on anomalies set:",vanilla_model.evaluate(x_test_anomalies,x_test_anomalies)) 



predicted=vanilla_model.predict(x_test) 
  #Compute MSE
r_error  = [np.square(x_test[i] - predicted[i]).mean() for i in range(len(x_test))]
if r_error[number] <= 0.05:
 plt.figure(figsize=(3,3))
 test_fig= px.imshow(x_test[number].reshape(28, 28), aspect='equal')
 plt.figure(figsize=(3,3))
 predicted_fig= px.imshow(predicted[number].reshape(28, 28), aspect='equal')
 plt.figure(figsize=(3,3))
 error_fig= px.imshow((np.square(x_test[number] - predicted[number])).reshape(28, 28), aspect='equal')
 st.text('Input:')
 st.plotly_chart(test_fig)
 st.text('Predicted:')
 st.plotly_chart(predicted_fig)
 st.text('Error:')
 st.plotly_chart(error_fig)
 st.write("Error on this element", r_error[number])
else:
 plt.figure(figsize=(3,3))
 test_fig= px.imshow(x_test[number].reshape(28, 28), aspect='equal')
 st.text('Input:')
 st.plotly_chart(test_fig)
 st.text('Error is greater then 5 %. It is a number?')
 positive_answer = st.checkbox('yes')
 negative_answer = st.checkbox('no')
 if positive_answer:
  plt.figure(figsize=(3,3))
  predicted_fig= px.imshow(predicted[number].reshape(28, 28), aspect='equal')
  plt.figure(figsize=(3,3))
  error_fig= px.imshow((np.square(x_test[number] - predicted[number])).reshape(28, 28), aspect='equal')
  st.text('Predicted:')
  st.plotly_chart(predicted_fig)
  st.text('Error:')
  st.plotly_chart(error_fig)
  st.write("Error on this element", r_error[number])
 elif negative_answer:
  st.text('It is an anomalie')
  st.write("Error on this element", r_error[number])
