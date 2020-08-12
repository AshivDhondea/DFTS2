"""
Utilities needed by the Data Generator.

Documentation to be sorted out later.

"""
#from PIL import Image # imported but unused.
import numpy as np
import os
#import keras # hans 10 july 2020
#from tensorflow.keras.preprocessing import image # edited. hans. 17 June 2020.

# hans 22 June 2020
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf

def absoluteFilePaths(directory):
	"""Finds the absolute paths of all images in the given directory
	"""
	dirList = os.listdir(directory)
	dirList = [os.path.join(directory, i) for i in dirList]
	dirList = [os.path.abspath(i) for i in dirList]
	return dirList

def pTasks(testImage, reshapeDims, norm):
	"""Performs image preprocessing used in the classification task

	# Arguments
		testImage: list containing the image label and path
		reshapeDims: reshape dimensions of the image
		norm: boolean representing whether the image need to be normalized

	# Returns
		Preprocessed image
	"""
	I = tf.keras.preprocessing.image.load_img(testImage[1])
	if reshapeDims != (-1, ):
		I = I.resize(reshapeDims)
	I = tf.keras.preprocessing.image.img_to_array(I)

	if norm:
		I = preprocess_input(I)
	return I

def _preprocess_numpy_input(x, data_format, mode):
	"""Preprocesses a Numpy array encoding a batch of images.
	# Arguments
		x: Input array, 3D or 4D.
		data_format: Data format of the image array.
		mode: One of "caffe", "tf" or "torch".
			- caffe: will convert the images from RGB to BGR,
				then will zero-center each color channel with
				respect to the ImageNet dataset,
				without scaling.
			- tf: will scale pixels between -1 and 1,
				sample-wise.
			- torch: will scale pixels between 0 and 1 and then
				will normalize each channel with respect to the
				ImageNet dataset.
	# Returns
		Preprocessed Numpy array.
	"""
	if not issubclass(x.dtype.type, np.floating):
		x = x.astype(backend.floatx(), copy=False)

	if mode == 'tf':
		x /= 127.5
		x -= 1.
		return x

	if mode == 'torch':
		x /= 255.
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
	else:
		if data_format == 'channels_first':
			# 'RGB'->'BGR'
			if x.ndim == 3:
				x = x[::-1, ...]
			else:
				x = x[:, ::-1, ...]
		else:
			# 'RGB'->'BGR'
			x = x[..., ::-1]
		mean = [103.939, 116.779, 123.68]
		std = None

	# Zero-center by mean pixel
	if data_format == 'channels_first':
		if x.ndim == 3:
			x[0, :, :] -= mean[0]
			x[1, :, :] -= mean[1]
			x[2, :, :] -= mean[2]
			if std is not None:
				x[0, :, :] /= std[0]
				x[1, :, :] /= std[1]
				x[2, :, :] /= std[2]
		else:
			x[:, 0, :, :] -= mean[0]
			x[:, 1, :, :] -= mean[1]
			x[:, 2, :, :] -= mean[2]
			if std is not None:
				x[:, 0, :, :] /= std[0]
				x[:, 1, :, :] /= std[1]
				x[:, 2, :, :] /= std[2]
	else:
		x[..., 0] -= mean[0]
		x[..., 1] -= mean[1]
		x[..., 2] -= mean[2]
		if std is not None:
			x[..., 0] /= std[0]
			x[..., 1] /= std[1]
			x[..., 2] /= std[2]
	return x

def preprocess_input(x, data_format='channels_last', mode='caffe'):
	"""Preprocesses a tensor or Numpy array encoding a batch of images.
	# Arguments
		x: Input Numpy or symbolic tensor, 3D or 4D.
			The preprocessed data is written over the input data
			if the data types are compatible. To avoid this
			behaviour, `numpy.copy(x)` can be used.
		data_format: Data format of the image tensor/array.
		mode: One of "caffe", "tf" or "torch".
			- caffe: will convert the images from RGB to BGR,
				then will zero-center each color channel with
				respect to the ImageNet dataset,
				without scaling.
			- tf: will scale pixels between -1 and 1,
				sample-wise.
			- torch: will scale pixels between 0 and 1 and then
				will normalize each channel with respect to the
				ImageNet dataset.
	# Returns
		Preprocessed tensor or Numpy array.
	# Raises
		ValueError: In case of unknown `data_format` argument.
	"""
	if data_format is None:
		data_format = backend.image_data_format()
	if data_format not in {'channels_first', 'channels_last'}:
		raise ValueError('Unknown data_format ' + str(data_format))

	if isinstance(x, np.ndarray):
		return _preprocess_numpy_input(x, data_format=data_format, mode=mode)
	else:
		return _preprocess_symbolic_input(x, data_format=data_format,
mode=mode)

