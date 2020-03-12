import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.layers import Conv1D, MaxPool1D, UpSampling1D, Input

import numpy as np

def build_cnn_model(parameters):
	window_size = parameters["window_size"]
	num_filters = parameters["filter_num"]
	filter_size = parameters["filter_size"]
	num_conv = parameters["num_conv"]
	reduced_model = parameters["reduced_model"]

	encoder_input = Input((window_size, 1)) # window_size, 1
	decoder_output = None
	latent = None

	if reduced_model:
		b_1 = build_block_down(encoder_input, num_filters, filter_size, num_conv)
		b_2 = build_block_down(b_1, num_filters, filter_size, num_conv)
		b_3 = build_block_down(b_2, num_filters, filter_size, num_conv)
		latent = Conv1D(num_filters, filter_size, padding="same", activation="relu", name="latent_space")(b_3)  
		b_4 = UpSampling1D(2)(latent) 
		b_5 = build_block_up(b_4, num_filters, filter_size, num_conv)
		b_6 = build_block_up(b_5, num_filters, filter_size, num_conv)
		decoder_output = Conv1D(1, filter_size, padding="same", activation="sigmoid", name="decoder_output")(b_6)
	else:
		## Downwards pass
		# conv_1 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(encoder_input)  
		# down_1 = MaxPool1D(2)(conv_1) # 512
		b_1 = build_block_down(encoder_input, num_filters, filter_size, num_conv)

		# conv_2 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_1)  
		# down_2 = MaxPool1D(2)(conv_2) # 256
		b_2 = build_block_down(b_1, num_filters, filter_size, num_conv)

		# conv_3 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_2)  
		# down_3 = MaxPool1D(2)(conv_3) # 128
		b_3 = build_block_down(b_2, num_filters, filter_size, num_conv)

		# conv_4 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_3)  
		# down_4 = MaxPool1D(2)(conv_4) # 64
		b_4 = build_block_down(b_3, num_filters, filter_size, num_conv)

		# conv_5 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_4)  
		# down_5 = MaxPool1D(2)(conv_5) # 32
		b_5 = build_block_down(b_4, num_filters, filter_size, num_conv)

		# conv_6 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_5)  
		# down_6 = MaxPool1D(2)(conv_6) # 16
		b_6 = build_block_down(b_5, num_filters, filter_size, num_conv)

		## Latent space
		latent = Conv1D(num_filters, filter_size, padding="same", activation="relu", name="latent_space")(b_6)  
		b_7 = UpSampling1D(2)(latent) # 32

		# conv_8 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_1)  
		# up_2 = UpSampling1D(2)(conv_8) # 64
		b_8 = build_block_up(b_7, num_filters, filter_size, num_conv)

		# conv_9 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_2)  
		# up_3 = UpSampling1D(2)(conv_9) # 128
		b_9 = build_block_up(b_8, num_filters, filter_size, num_conv)

		# conv_10 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_3)  
		# up_4 = UpSampling1D(2)(conv_10) # 256
		b_10 = build_block_up(b_9, num_filters, filter_size, num_conv)

		# conv_11 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_4)  
		# up_5 = UpSampling1D(2)(conv_11) # 512
		b_11 = build_block_up(b_10, num_filters, filter_size, num_conv)

		# conv_12 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_5)  
		# up_6 = UpSampling1D(2)(conv_12) # 1024
		b_12 = build_block_up(b_11, num_filters, filter_size, num_conv)

		decoder_output = Conv1D(1, filter_size, padding="same", activation="sigmoid", name="decoder_output")(b_12)

	model = keras.Model(inputs=[encoder_input], outputs=[decoder_output, latent])

	losses = {"decoder_output" : "mse"}

	model.compile(optimizer=keras.optimizers.Adam(lr=0.002), loss=losses, metrics=[])
	print(model.summary())

	return model

def build_block_down(input, num_filters, filter_size, num_conv=1, name=None):
	current = input
	for _ in range(num_conv):
		current = Conv1D(num_filters, filter_size, padding="same", activation="relu", name=name)(current)
	return MaxPool1D(2)(current)

def build_block_up(input, num_filters, filter_size, num_conv=1, name=None):
	current = input
	for _ in range(num_conv):
		current = Conv1D(num_filters, filter_size, padding="same", activation="relu", name=name)(current)
	return UpSampling1D(2)(current) 

def build_cnn_model_test(parameters, weights_file):
	window_size = parameters["window_size"]
	num_filters = parameters["filter_num"]
	filter_size = parameters["filter_size"]
	num_conv = parameters["num_conv"]
	reduced_model = parameters["reduced_model"]
	
	encoder_input = Input((window_size, 1)) # 1024, 1
	decoder_output = None
	latent = None
	
	if reduced_model:
		b_1 = build_block_down(encoder_input, num_filters, filter_size, num_conv)
		b_2 = build_block_down(b_1, num_filters, filter_size, num_conv)
		b_3 = build_block_down(b_2, num_filters, filter_size, num_conv)
		latent = Conv1D(num_filters, filter_size, padding="same", activation="relu", name="latent_space")(b_3)  
		b_4 = UpSampling1D(2)(latent) 
		b_5 = build_block_up(b_4, num_filters, filter_size, num_conv)
		b_6 = build_block_up(b_5, num_filters, filter_size, num_conv)
		decoder_output = Conv1D(1, filter_size, padding="same", activation="sigmoid", name="decoder_output")(b_6)
	else:
		## Downwards pass
		# conv_1 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(encoder_input)  
		# down_1 = MaxPool1D(2)(conv_1) # 512
		b_1 = build_block_down(encoder_input, num_filters, filter_size, num_conv)

		# conv_2 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_1)  
		# down_2 = MaxPool1D(2)(conv_2) # 256
		b_2 = build_block_down(b_1, num_filters, filter_size, num_conv)

		# conv_3 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_2)  
		# down_3 = MaxPool1D(2)(conv_3) # 128
		b_3 = build_block_down(b_2, num_filters, filter_size, num_conv)

		# conv_4 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_3)  
		# down_4 = MaxPool1D(2)(conv_4) # 64
		b_4 = build_block_down(b_3, num_filters, filter_size, num_conv)

		# conv_5 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_4)  
		# down_5 = MaxPool1D(2)(conv_5) # 32
		b_5 = build_block_down(b_4, num_filters, filter_size, num_conv)

		# conv_6 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(down_5)  
		# down_6 = MaxPool1D(2)(conv_6) # 16
		b_6 = build_block_down(b_5, num_filters, filter_size, num_conv)

		## Latent space
		latent = Conv1D(num_filters, filter_size, padding="same", activation="relu", name="latent_space")(b_6)  
		b_7 = UpSampling1D(2)(latent) # 32

		# conv_8 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_1)  
		# up_2 = UpSampling1D(2)(conv_8) # 64
		b_8 = build_block_up(b_7, num_filters, filter_size, num_conv)

		# conv_9 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_2)  
		# up_3 = UpSampling1D(2)(conv_9) # 128
		b_9 = build_block_up(b_8, num_filters, filter_size, num_conv)

		# conv_10 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_3)  
		# up_4 = UpSampling1D(2)(conv_10) # 256
		b_10 = build_block_up(b_9, num_filters, filter_size, num_conv)

		# conv_11 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_4)  
		# up_5 = UpSampling1D(2)(conv_11) # 512
		b_11 = build_block_up(b_10, num_filters, filter_size, num_conv)

		# conv_12 = Conv1D(num_filters, filter_size, padding="same", activation="relu")(up_5)  
		# up_6 = UpSampling1D(2)(conv_12) # 1024
		b_12 = build_block_up(b_11, num_filters, filter_size, num_conv)

		decoder_output = Conv1D(1, filter_size, padding="same", activation="sigmoid", name="decoder_output")(b_12)

	model = keras.Model(inputs=[encoder_input], outputs=[decoder_output, latent])

	model.compile(optimizer=keras.optimizers.Adam(lr=0.002), loss="mse", metrics=[])
	print(model.summary())

	model.load_weights(weights_file)

	return model
