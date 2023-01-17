import streamlit as st 
#st.markdown(""" This is a Streamlit App """)
import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf

#TO DO: get the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()



mT = x_train.mean()
sT = x_train.std()
x_test_anomalies =[np.random.normal(mT,sT,784).reshape(28,28)]

for x in range(0,500):
  new_x=np.random.normal(mT,sT,784).reshape(28,28)
  x_test_anomalies=np.append(x_test_anomalies, [new_x], axis=0)






mT = x_train.mean()
sT = x_train.std()
x_test_anomalies =[np.random.normal(mT,sT,784).reshape(28,28)]

for x in range(0,500):
  new_x=np.random.normal(mT,sT,784).reshape(28,28) +x_test[x+1].reshape(28,28)+x_test[x+3].reshape(28,28)
  x_test_anomalies=np.append(x_test_anomalies, [new_x], axis=0) #+x_test[x+3].reshape(28,28)




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
option_func = ['relu', 'tanh', 'softmax', 'sigmoid']
actfun = st.sidebar.selectbox('Which activation function do you like to use?', option_func)
number_of_nodes = st.sidebar.select_slider('number of nodes', options=(2,4,8,16))
if number_of_nodes == 2:
  number_of_layers = st.sidebar.select_slider('number of layers', options=(2,4,6,8))
elif number_of_nodes == 4:
  number_of_layers = st.sidebar.select_slider('number of layers', options=(2,4))
else:
  st.text('Only 2 layers possible with your number of nodes selection. Do you want to continue?')
  continue_with_model = st.checkbox('yes')
  back_to_selection = st.checkbox('no')
  if continue_with_model:
    number_of_layers = 2
    st.text('Press RUN to continue')
  elif back_to_selection:
    st.text('Select other number of nodes')
if st.button('RUN'):
 vanilla_model= model_def(number_of_nodes,number_of_layers, actfun)

 vanilla_model.summary()

 batch_size = 128
 epochs = 15

# TO DO: Compile the model
 vanilla_model.compile(optimizer='adam', loss='mean_squared_error') #MAE
 history = vanilla_model.fit(x_train, x_train, batch_size= batch_size, epochs = epochs, validation_split= 0.1) 


 predicted=vanilla_model.predict(x_train[3:4])


 r = vanilla_model.predict(x_test)   


  # TO DO : Compute MSE
 r_error  = [np.square(x_test[i] - r[i]).mean() for i in range(len(x_test))]







 r_anomalities = vanilla_model.predict(x_test_anomalies)   


  # TO DO : Compute MSE
 r_error  = [np.square(x_test_anomalies[i] - r_anomalities[i]).mean() for i in range(len(x_test_anomalies))]








 predicted_test = vanilla_model.predict(x_test)
 train_loss = tf.keras.losses.mae(predicted_test, x_test)

 r_anomalities_test= vanilla_model.predict(x_test_anomalies)
 train_loss=tf.keras.losses.mae(r_anomalities_test, x_test_anomalies)
