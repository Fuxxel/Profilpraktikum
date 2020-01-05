import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

if tf.__version__[0] == "1": # TF version 1.x.x
	if tf.test.gpu_device_name():
		from tensorflow.keras.layers import CuDNNLSTM as LSTM
	else:
		from tensorflow.keras.layers import LSTM as LSTM
else:
	from tensorflow.keras.layers import LSTM as LSTM

import numpy as np

def build_enc_dec_model(parameters):
	latent_dim = parameters["latent_dim"]

	# Define an input sequence and process it.
	encoder_inputs = keras.Input(shape=(None, 1), name="encoder_inputs")
	_, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm")(encoder_inputs)

	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = keras.Input(shape=(None, 1), name="decoder_inputs")

	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the
	# return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='linear'))
	decoder_outputs = decoder_dense(decoder_outputs)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])

	model.compile(optimizer=keras.optimizers.Adam(lr=0.002), loss="mse", metrics=[])
	print(model.summary())

	return model
