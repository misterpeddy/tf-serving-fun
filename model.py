import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt

tf.logging.set_verbosity('DEBUG')

module_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
image_path = "cat.jpg"

with tf.Graph().as_default():
  model = hub.Module(module_url)
  image_string_placeholder = tf.placeholder(tf.string)
  decoded_image = tf.image.decode_jpeg(image_string_placeholder)
  decoded_image_float = tf.image.convert_image_dtype(
      image=decoded_image, dtype=tf.float32)
  module_input = tf.expand_dims(decoded_image_float, 0)
  result = model(module_input, as_dict=True)
  init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

  session = tf.Session()
  session.run(init_ops)

  with tf.gfile.Open(image_path, "rb") as binfile:
    image_string = binfile.read()

  result_out, image_out = session.run(
      [result, decoded_image],
      feed_dict={image_string_placeholder: image_string})
  print("Found %d objects." %len(result_out["detection_scores"]))

image = np.array(image_out)

fig = plt.figure(figsize=(20,15))
plt.grid(False)
plt.imshow(image)
