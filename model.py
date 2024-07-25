from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.layers import Layer
from keras import backend as K
from keras.models import Sequential
from keras.layers import InputLayer

class Onehot(Layer):
	def __init__(self, output_dim,**kwargs):
		self.output_dim = int(output_dim)
		kwargs['input_dtype'] = 'int32'
		super(Onehot,self).__init__(**kwargs)

	def build(self, input_shape):
		self.in_shape = input_shape
		self.built = True

	def call(self, x):
		return K.one_hot(x, self.output_dim)

	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim)


def buildNetwork(state_size, action_size, bins = 51, lr = 0.00025, dropout_keep = 1):
	input = Input(shape = ([state_size]))		
	# hidden = Dense(256, activation = 'relu')(input)
	# hidden1 = Dense(128, activation = 'relu')(hidden)
	# hidden = Dense(128, activation = 'relu')(hidden)
	output = []
	for i in range(action_size):
		hidden = Dense(128, activation = 'relu')(input)
		hidden1 = Dense(128, activation = 'relu')(hidden)
		# hidden1 = Dense(128, activation = 'relu')(hidden)
		# if dropout_keep != 1:
		# 	hidden = Dropout(dropout_keep)(hidden)
		Layer = Dense(bins, activation = 'softmax')(hidden1)
		# if dropout_keep != 1:
		# 	Layer = Dropout(dropout_keep)(Layer)
		output.append(Layer)

	model = Model(input = input, output =  output)
	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = lr))
	return model



def buildSingleNetwork(state_size, action_size, bins = 51, lr = 0.00025, p_keep = 1, discrete = 0):

	model = Sequential()
	if discrete:
		model.add(InputLayer(input_shape = (1,), dtype = 'int32'))
		model.add(Onehot(state_size))
	else:
		model.add(InputLayer(input_shape  = ([state_size])))

	# model.add(Dense(256,activation = 'relu'))

	# if p_keep!=1:
	# 	model.add(Dropout(1-p_keep))
	# model.add(Dense(256,activation = 'relu'))

	if p_keep!=1:
		model.add(Dropout(1-p_keep))
	model.add(Dense(bins, activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = lr))
	return model

	# inputs = Input(shape = ([state_size]))
	# if p_keep != 1:
	# 	hidden = Dropout(1-p_keep)(inputs,  training=True)
	# 	hidden = Dense(256, activation = 'relu')(hidden)
	# else:
	# 	hidden = Dense(256, activation = 'relu')(inputs)
	# if p_keep != 1:
	# 	hidden = Dropout(1-p_keep)(hidden, training=True)

	# hidden = Dense(256, activation = 'relu')(hidden)
	# if p_keep != 1:
	# 	hidden = Dropout(1-p_keep)(hidden, training=True)

	# # hidden = Dense(256, activation = 'relu')(hidden)
	# # if p_keep != 1:
	# # 	hidden = Dropout(1-p_keep)(hidden, training=True)

	# Layer = Dense(bins, activation = 'softmax')(hidden)
	# # if dropout_keep != 1:
	# # 	Layer = Dropout(dropout_keep)(Layer)
	# output = Layer

	# model = Model(input = inputs, output =  output)
	# model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = lr))
	# return model