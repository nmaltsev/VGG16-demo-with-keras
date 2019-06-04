import cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions                        
from keras.preprocessing.image import load_img, img_to_array  
from keras.models import Model

import tensorflow as tf
from keras import backend as K

num_cores = 4
num_CPU = 1
num_GPU = 0

gpu_options = tf.GPUOptions(
	per_process_gpu_memory_fraction=0.9,
	allow_growth=True
)
# FIX
config = tf.ConfigProto(
	gpu_options=gpu_options,
	intra_op_parallelism_threads=num_cores,
	inter_op_parallelism_threads=num_cores, 
	allow_soft_placement=True,
	device_count = {'CPU': num_CPU,'GPU': num_GPU}
)

session = tf.Session(config=config)
K.set_session(session)
K.set_image_dim_ordering('tf')

def readImage(image_path):
	img = load_img(image_path, target_size=(224, 224))
	img = img_to_array(img)
	return img
		
def readImageCv2(image_path):		
	img = cv2.imread(image_path).astype('float32')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (224,224))
	return img
		
def classificate_images(images, is_debug=False):
	model = VGG16(weights='imagenet', include_top=True)
	if is_debug: print(model.summary())
	
	
	for image_path in images:
		image = readImage(image_path)
		# reshape the data
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		print('Image %s ready' % (image_path))

		model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)
		
		fc2_features = model_extractfeatures.predict(image)
		fc2_features_2 = fc2_features.reshape((4096,1))
		pic=features[0,:,:,128]
		print(fc2_features_2)
		# np.savetxt('fc2.txt',fc2_features)
		# TODO

if __name__ == "__main__":
	classificate_images([
		'images/9999.jpg',
		'images/1008.jpg'
	])

