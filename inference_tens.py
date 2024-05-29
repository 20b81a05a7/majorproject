import os
import tensorflow as tf

import matplotlib.image as mpimg
import numpy as np
import PIL.Image

import tensorflow_hub as hub
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

class Master:

    def __init__(self) -> None:
        self.hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')



    def tensor_to_image(self,tensor):
       tensor = tensor*255
       tensor = np.array(tensor, dtype=np.uint8)
       if np.ndim(tensor)>3:
          assert tensor.shape[0] == 1
          tensor = tensor[0]
       PIL.Image.fromarray(tensor).save('style.jpg')
    
    def load_img(self,path_to_img):
       max_dim = 512
       img = tf.io.read_file(path_to_img)
       img = tf.image.decode_image(img, channels=3)
       img = tf.image.convert_image_dtype(img, tf.float32)

       shape = tf.cast(tf.shape(img)[:-1], tf.float32)
       long_dim = max(shape)
       scale = max_dim / long_dim

       new_shape = tf.cast(shape * scale, tf.int32)

       img = tf.image.resize(img, new_shape)
       img = img[tf.newaxis, :]
       return img
    
 
    def put_to(self,main_addr,style_addr):

        content = self.load_img(main_addr)
        style = self.load_img(style_addr) 
        stylized_image = self.hub_model(tf.constant(content), tf.constant(style))[0]
        self.tensor_to_image(stylized_image)

#m = Master()
#m.put_to("data/input_image/RKS.jpg","data/target_style/dotted_style.jpg")

 