import tensorflow as tf

# 28x28 입력, 10가지 결과
x = tf.placeholder("float", [None, 28,28],name='x')
y_ = tf.placeholder("float", [None,10],name='y_')
keep_prob = tf.placeholder("float", name='keep_prob')

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# convolution layer
def lenet5_layer(input, weight, bias,weight_name,bias_name):
    W_conv = weight_variable(weight,weight_name)
    b_conv = bias_variable(bias,bias_name)
    h_conv = tf.nn.relu(conv2d(input, W_conv) + b_conv)
    return max_pool_2x2(h_conv)

# connected layer
def dense_layer(layer, weight, bias,weight_name,bias_name):
    W_fc = weight_variable(weight,weight_name)
    b_fc = bias_variable(bias,bias_name)
    return tf.nn.relu(tf.matmul(layer, W_fc) + b_fc)

def build_model(is_training):
    #first conv
    x_image = tf.reshape(x, [-1,28,28,1])
    W_conv1 = [5, 5, 1, 32]
    b_conv1 = [32]
    layer1 = lenet5_layer(x_image,W_conv1,b_conv1,'W_conv1','b_conv1')
    #second conv
    W_conv2 = [5, 5, 32, 64]
    b_conv2 = [64]
    layer2 = lenet5_layer(layer1,W_conv2,b_conv2,'W_conv2','b_conv2')
    #third conv
    W_fc1 = [7 * 7 * 64, 1024]
    b_fc1 = [1024]
    layer2_flat = tf.reshape(layer2, [-1, 7*7*64])
    layer3 = dense_layer(layer2_flat,W_fc1,b_fc1,'W_fc1','b_fc1')
    #softmax
    W_fc2 = weight_variable([1024, 10],'W_fc2')
    b_fc2 = bias_variable([10],'b_fc2')
    if is_training:
        #dropout
        h_fc1_drop = tf.nn.dropout(layer3, keep_prob)
        finaloutput=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="y_conv")
    else:
        finaloutput=tf.nn.softmax(tf.matmul(layer3, W_fc2) + b_fc2,name="y_conv")
    print('finaloutput:', finaloutput)
    return finaloutput