import tensorflow as tf
import numpy as np

y=tf.Variable(0,name='y')

model=tf.global_variables_initializer()

with tf.Session() as sess:
    for i in range(5):
        x = np.random.randint(1000)
        print(int(x))
        sess.run(model)
        y=(y+x)/(i+1)
        print(sess.run(y))
