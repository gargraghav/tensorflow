import tensorflow as tf

x = tf.constant(35, name='x')
print(x)
y = tf.Variable(x + 5, name='y')

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/home/raghav/PycharmProjects/tensorflow/tensorboard/output5", session.graph)
    model =  tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))
    writer.close()