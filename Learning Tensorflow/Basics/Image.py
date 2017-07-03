import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from os.path import dirname, realpath, join

dir=join(dirname(realpath(dirname(realpath(dirname(__file__))))),'Images')
filename=dir + "/MarshOrchid.jpg"
img_data=mpimg.imread(filename)

image=tf.placeholder("uint8",[None,None,3])
slice=tf.slice(image,[1000,0,0],[3000,-1,-1])

with tf.Session() as sess:
    result=sess.run(slice, feed_dict={image:img_data})
    print(result.shape)

plt.imshow(result)
plt.show()
