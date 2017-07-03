import tensorflow as tf
import numpy as np

x=np.random.randint(100, size=25)
y=tf.Variable(5*(x**2)- 3*x + 15,name='y')

model=tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))

