import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train=mnist.train.images
Y_train=mnist.train.images
X_test=mnist.test.images
Y_test=mnist.test.labels

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

id=36436
img=np.reshape(X_train[id,:],(28,28))

#plt.matshow(img)
#plt.show()

X=tf.placeholder(tf.float32,[None,784],name="input")
X=tf.placeholder(tf.float32,[None,10],name="output")
W=tf.Variable(tf.zeros([784,10]),name="weight")
b=tf.Variable(tf.zeros([10]),name="bias")

Y_pred=tf.nn.softmax(tf.matmul(X,W)+b)

loss=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(Y_pred),reduction_indices=1))

learning_rate=0.005
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
training_epochs=50
display_epoch=5
batch_size=100

correct_prediction=tf.equal(tf.argmax(Y_pred,1),tf.argmax(Y,1))
accuracy= tf.reduce_mean(tf.cast(correct_prediction,"float"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(training_epochs):
        nBatch=int(55000/batch_size)
        id1=np.random.permutation(55000)
        for i in range(nBatch):
            X_batch=X_train[id1[i*batch_size:(i+1)*batch_size],:]
            Y_batch = Y_train[id1[i*batch_size:(i + 1) * batch_size],:]
            sess.run(optimizer, feed_dict={X:X_batch, Y:Y_batch})

        if (epoch+1) % display_epoch == 0:
            loss_temp=sess.run(loss,feed_dict={X:X_train,Y:Y_train})
            accuracy_temp=accuracy.eval({X:X_train, Y: Y_train})
            print("(epoch{})".format(epoch+1))
            print("[Loss/Training Accuracy] {:05.4f} / {:05.4f}".format(loss_temp,accuracy_temp))
            print(" ")
    print ("[Test Accuracy]", accuracy.eval({X:X_test,Y: Y_test}))


