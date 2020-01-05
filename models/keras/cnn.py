import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.layers import Conv1D, MaxPool1D, UpSampling1D, Input

import numpy as np

def build_cnn_model(parameters):
	window_size = parameters["window_size"]
	num_filters = parameters["filter_num"]
	
	encoder_input = Input((window_size, 1)) # window_size, 1

	## Downwards pass
	conv_1 = Conv1D(num_filters, 3, padding="same", activation="relu")(encoder_input)  
	down_1 = MaxPool1D(2)(conv_1) # /2

	conv_2 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_1)  
	down_2 = MaxPool1D(2)(conv_2) # /4

	conv_3 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_2)  
	down_3 = MaxPool1D(2)(conv_3) # /8

	conv_4 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_3)  
	down_4 = MaxPool1D(2)(conv_4) # /16

	conv_5 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_4)  
	down_5 = MaxPool1D(2)(conv_5) # /32

	conv_6 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_5)  
	down_6 = MaxPool1D(2)(conv_6) # /64 

	## Upwards pass
	conv_7 = Conv1D(num_filters, 3, padding="same", activation="relu", name="latent_space")(down_6)  # --> latent space
	up_1 = UpSampling1D(2)(conv_7) # /32
 
	conv_8 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_1)  
	up_2 = UpSampling1D(2)(conv_8) # /16

	conv_9 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_2)  
	up_3 = UpSampling1D(2)(conv_9) # /8

	conv_10 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_3)  
	up_4 = UpSampling1D(2)(conv_10) # /4

	conv_11 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_4)  
	up_5 = UpSampling1D(2)(conv_11) # /2

	conv_12 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_5)  
	up_6 = UpSampling1D(2)(conv_12) # window_size

	decoder_output = Conv1D(1, 3, padding="same", activation="sigmoid", name="decoder_output")(up_6)

	model = keras.Model(inputs=[encoder_input], outputs=[decoder_output, conv_7])

	losses = {"decoder_output" : "mse"}

	model.compile(optimizer=keras.optimizers.Adam(lr=0.002), loss=losses, metrics=[])
	print(model.summary())

	return model

def build_cnn_model_test(parameters, weights_file):
	window_size = parameters["window_size"]
	num_filters = parameters["filter_num"]
	
	encoder_input = Input((window_size, 1)) # 1024, 1

	## Downwards pass
	conv_1 = Conv1D(num_filters, 3, padding="same", activation="relu")(encoder_input)  
	down_1 = MaxPool1D(2)(conv_1) # 512

	conv_2 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_1)  
	down_2 = MaxPool1D(2)(conv_2) # 256

	conv_3 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_2)  
	down_3 = MaxPool1D(2)(conv_3) # 128

	conv_4 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_3)  
	down_4 = MaxPool1D(2)(conv_4) # 64

	conv_5 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_4)  
	down_5 = MaxPool1D(2)(conv_5) # 32

	conv_6 = Conv1D(num_filters, 3, padding="same", activation="relu")(down_5)  
	down_6 = MaxPool1D(2)(conv_6) # 16

	## Upwards pass
	conv_7 = Conv1D(num_filters, 3, padding="same", activation="relu", name="latent_space")(down_6)  
	up_1 = UpSampling1D(2)(conv_7) # 32

	conv_8 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_1)  
	up_2 = UpSampling1D(2)(conv_8) # 64

	conv_9 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_2)  
	up_3 = UpSampling1D(2)(conv_9) # 128

	conv_10 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_3)  
	up_4 = UpSampling1D(2)(conv_10) # 256

	conv_11 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_4)  
	up_5 = UpSampling1D(2)(conv_11) # 512

	conv_12 = Conv1D(num_filters, 3, padding="same", activation="relu")(up_5)  
	up_6 = UpSampling1D(2)(conv_12) # 1024

	decoder_output = Conv1D(1, 3, padding="same", activation="sigmoid", name="decoder_output")(up_6)

	model = keras.Model(inputs=[encoder_input], outputs=[decoder_output, conv_7])

	model.compile(optimizer=keras.optimizers.Adam(lr=0.002), loss="mse", metrics=[])
	print(model.summary())

	model.load_weights(weights_file)

	return model
