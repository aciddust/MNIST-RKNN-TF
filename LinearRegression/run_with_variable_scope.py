import tensorflow as tf
import numpy as np

# 초기화 : 데이터 불러오기
data = np.loadtxt('./dataset/data.txt', delimiter=',', unpack=True, dtype='float32')

x_data = data[:4] 
y_data = data[4:] 

print(x_data)
print(y_data)

with tf.variable_scope('test'):
    X = tf.placeholder(tf.float32, name='x')
    Y = tf.placeholder(tf.float32, name='y')

    ## 가중치와 바이어스 범위는 -100 ~ 100
    W = tf.Variable(tf.random_uniform([1], -100., 100.), name='w')
    b = tf.Variable(tf.random_uniform([1], -100., 100.), name='b')

    mul = tf.multiply(W, X, name='mul')
    hypothesis = tf.add(mul, b, name='hypothesis')

    ## hypothesis & cost func
    cost = tf.reduce_mean(tf.square(hypothesis - Y), name='cost')

    rate = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(rate)
    train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.compat.v1.Session()
sess.run(init)

op = sess.graph.get_operations()
for m in op:
    print(m.values())

saver = tf.compat.v1.train.Saver(max_to_keep=3)

for step in range(2002):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))

saver.save(sess, './output/linear_regression.ckpt')
tf.io.write_graph(sess.graph_def, '.', './output/linear_regression.pb', as_text=False)
tf.io.write_graph(sess.graph_def, '.', './output/linear_regression.pbtxt', as_text=True)

print('[TEST]')
print(sess.run(hypothesis, feed_dict={X: [1,2,3,4,5,6,7,8,9,10,20,21,22,23,24,25,40,41,42,43,44]}))

