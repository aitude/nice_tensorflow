"""Non-Linear Independent Components Estimation with Tensorflow

This script is implementation of Non-Linear Independent components estimation algorithm with Tensorflow

Command- python nice.py

This file contains the following class & functions:
	* ScalingLayer - This class rescales the output of coupling layers.
    * CouplingLayer - This class implement additive coupling layer.
    * NICE - This class train the model using NICE algorithm and generate new images.
"""

# Load Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Load Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


class ScalingLayer(layers.Layer):
	""" This method define the scaling layer with normal distribution.

	Parameters
	----------
	input_dim : int
	- Number of Pixels - width * height
	"""
	def __init__(self,input_dim):
		super(ScalingLayer,self).__init__()
		self.scaling_layer = tf.Variable(tf.random.normal([input_dim]),trainable=True)

	""" This method rescale the final out of coupling layers.

	Parameters
	----------
	x : Tensor Array
	- Tensors to be rescale.
	"""
	def call(self,x):
		return x*tf.transpose(tf.math.exp(self.scaling_layer))

	""" This method is inverse implementation of the call method.

	Parameters
	----------
	z : Tensor Array
	    - Tensors to be rescale.
	"""
	def inverse(self,z):
	    return z*tf.transpose(tf.math.exp(-self.scaling_layer))

class CouplingLayer(layers.Layer):

	""" This method define the neural network to learn function F

    Parameters
    ----------
    input_dim : int
        - Number of Pixels - width * height
    hidden_dim : int
        - Number of Units per hidden layer
    partition : String
        - Odd or even indexing scheme
    num_layers : int
        - Number of hidden layer
    
    """

	def __init__(self,input_dim, hidden_dim, partition, num_hidden_layers):
		
		super(CouplingLayer, self).__init__()

		assert input_dim % 2 == 0

		self.partition = partition # Posssible values - odd or even

		# Define Neural Network

		inputs = keras.Input(shape = (input_dim))
		
		x = inputs
		for layer in range(num_hidden_layers):
			x = layers.Dense(hidden_dim)(x)
			x = layers.LeakyReLU()(x)

		outputs = layers.Dense(input_dim, activation='linear')(x)
		self.m = keras.Model( inputs = inputs, outputs = outputs)

		# Define lambda functions to get odd or even indexed data.
		if self.partition == 'even':
			self._first = lambda xs: xs[:,0::2]
			self._second = lambda xs: xs[:,1::2]
		else:
			self._first = lambda xs: xs[:,1::2]
			self._second = lambda xs: xs[:,0::2]

	""" This method implement additive coupling layer operations. 
	y1 = x1
	y2 = x2 + m(x1)

    Parameters
    ----------
    inputs : Tensor Array
        Input Tensors.

    Returns
        -------
        Tensor Array
            Latent representation.
    """

	def call(self,inputs):

		# Split input into two parts x1 and x2
		x1 = self._first(inputs)
		x2 = self._second(inputs)
		
		# Inference latent representation
		y1 = x1
		y2 = x2 + self.m(x1)
		
		# Merge both y1 an y2 using interleave method. e.g y1 = [1,3,5,7,9] and y2 = [2,4,6,8,10] then y = [1,2,3,4,5,6,7,8,9,10]
		if self.partition == 'even':
		      y = tf.reshape(tf.concat([y1[...,tf.newaxis], y2[...,tf.newaxis]], axis=-1), [tf.shape(y1)[0],-1])
		else:
		      y = tf.reshape(tf.concat([y2[...,tf.newaxis], y1[...,tf.newaxis]], axis=-1), [tf.shape(y2)[0],-1])
		return y

	"""This function will generate new image using latent variable

    Parameters
    ----------
    latent_variable : Tensor Array
       . latent variable from logistic distribution.
    
    Returns
    -------
    List
        Pixels value.
    """
	def inverse(self,latent_variable):

		y1 = self._first(latent_variable)
		y2 = self._second(latent_variable)
		x1,x2 = y1, y2 - self.m(y1)

		if self.partition == 'even':
		      x = tf.reshape(tf.concat([x1[...,tf.newaxis], x2[...,tf.newaxis]], axis=-1), [tf.shape(x1)[0],-1])
		else:
		      x = tf.reshape(tf.concat([x2[...,tf.newaxis], x1[...,tf.newaxis]], axis=-1), [tf.shape(x2)[0],-1])

		return x

class NICE(keras.Model):

	""" This function define trainable model with multiple coupling layers and a scaling layer.

    Parameters
    ----------
    input_dim : int
        - Number of Pixels - width * height
    hidden_dim : int
        - Number of Units per hidden layer
    num_hidden_layers : int
        - Number of hidden layer
    
    """

	def __init__(self,input_dim,hidden_dim,num_hidden_layers):

		super(NICE,self).__init__()
		self.input_dim = input_dim
		# Coupling layer will have half dimension of the input data.
		half_dim = int(self.input_dim / 2)

		# Define 4 coupling layers.
		self.coupling_layer1 = CouplingLayer(input_dim=half_dim, hidden_dim = hidden_dim, partition='odd', num_hidden_layers=num_hidden_layers)
		self.coupling_layer2 = CouplingLayer(input_dim=half_dim, hidden_dim = hidden_dim, partition='even', num_hidden_layers=num_hidden_layers)
		self.coupling_layer3 = CouplingLayer(input_dim=half_dim, hidden_dim = hidden_dim, partition='odd', num_hidden_layers=num_hidden_layers)
		self.coupling_layer4 = CouplingLayer(input_dim=half_dim, hidden_dim = hidden_dim, partition='even', num_hidden_layers=num_hidden_layers)
		
		# Define scaling layer which rescaling the output for more weight variations.
		self.scaling_layer = ScalingLayer(self.input_dim)

	"""This function calculates the log likelihood.

    Parameters
    ----------
    z : Tensor Array
        Latent space representation.

    Returns
        -------
        List
            Log likelihood.
    """

	def log_likelihood(self, z):
		log_likelihood = tf.reduce_sum(-(tf.math.softplus(z) + tf.math.softplus(-z)), axis=1) + tf.reduce_sum(self.scaling_layer.scaling_layer)
		return log_likelihood

	""" This function passes input through coupling layer. Scale the output using scaling layer and return latent space z and loglikelihood

    Parameters
    ----------
    x : Tensor Array
        Input data from training samples.

    Returns
        -------
        Tensor Array
            Latent space representation.
        List
        	Log-likelihood for each image in the batch
    """

	def call(self, x):
		
		z = self.coupling_layer1(x)
		z = self.coupling_layer2(z)
		z = self.coupling_layer3(z)
		z = self.coupling_layer4(z)
		z = self.scaling_layer(z)

		log_likelihood = self.log_likelihood(z)
		return z,log_likelihood

	""" Generate sample data using latent space. This is exact inverse of the call method.

    Parameters
    ----------
    z : Tensor Array
        Latent Spaec.

    Returns
    -------
    Tensor Array
        Newly generated data
    """

	def inverse(self,z):
		x  = z
		x  = self.scaling_layer.inverse(x)
		x  = self.coupling_layer4.inverse(x)
		x  = self.coupling_layer3.inverse(x)
		x  = self.coupling_layer2.inverse(x)
		x  = self.coupling_layer1.inverse(x)

		return x

	""" Sample out new data from learnt probability distribution

    Parameters
    ----------
    num_samples : int
        - Number of sample images to generate.

    Returns
    -------
    List
    	- List of generated sample data.
        
    """

	def sample(self, num_samples):
		z = tf.random.uniform([num_samples, self.input_dim])
		z = tf.math.log(z) - tf.math.log(1.-z)
		return self.inverse(z)

""" This function show the generated images """
def show_samples():
    # Generate 100 sample digits
    ys = model.sample(21)
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({
        "grid.color": "black",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})
    new_images = tf.reshape(ys,(-1,28,28))
    for i in range(1,20):
      plt.subplot(2,10,i)
      plt.axis('off')
      plt.imshow(new_images[i],cmap="gray")
    plt.show()

# Basic Configuration
input_data_size = 784 # for mnist dataset. 28*28
num_epochs = 1
batch_size = 256
num_hidden_layer = 5
num_hidden_units = 1000

# Prepare training dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.reshape(-1,input_data_size) + tf.random.uniform([input_data_size])) / 256.0
dataloader = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# Custom Training Loop for the NICE Model
model = NICE(input_dim = input_data_size, hidden_dim = num_hidden_units, num_hidden_layers = num_hidden_layer)

opt = keras.optimizers.Adam(learning_rate=0.001,clipnorm=1.0)

# Iterate through training epoch
for i in range(num_epochs):

	# Iterate through dataloader
	for batch_id,(x,_) in enumerate(dataloader):

		# Forward Pass
		with tf.GradientTape() as tape:
			z, likelihood = model(x)
			log_likelihood_loss = tf.reduce_mean(likelihood)
			# Convert the maximum likelihood problem into negative of minimum likelihood problem
			loss = -log_likelihood_loss
		# Backward Pass
		gradient = tape.gradient(loss,model.trainable_weights)
		opt.apply_gradients(zip(gradient,model.trainable_weights))

	# Log the maxmimum likelihood after each training loop
	print('Epoch {} completed. Log Likelihood:{}'.format(i,log_likelihood_loss))
	# Generate 20 new images after 10 epochs.
	if i % 10 == 0 and i!=0:
		show_samples()

model.summary()
# Save model
model.save("nicegenerative")