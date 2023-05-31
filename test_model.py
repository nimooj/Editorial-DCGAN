import os
from numpy import asarray
from numpy.random import randn, randint
from keras.models import load_model
from matplotlib import pyplot
import tensorflow as tf


model_path = 'models/'
model_name = 'EditorialDCGAN.h5'
latent_dim = 200

def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)

	return z_input

def plot_generated(examples, n):
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i])

    pyplot.show()


model = load_model(model_path + model_name)
latents = generate_latent_points(latent_dim, 400)

results = model.predict(latents)
results = (results + 1) / 2.0

plot_generated(results, 10)
