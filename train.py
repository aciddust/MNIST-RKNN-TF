'''
여기서 create_training_graph()에 전달 된 is_quantify 매개 변수를 False로 설정.
mnist로부터 얻은 train 및 test 데이터 형태는 모두 (784,).
reshape_batch 함수는 train의 배치와 test의 입력을 (28,28)로 재구성하기 위해 정의되며,
구체적인 코드는 아래와 같음.
'''

# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import build_model, x, y_, keep_prob

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
def create_training_graph(is_quantify):
    # train 그래프 작성 후 create_training_graph 추가:
    g = tf.get_default_graph()   # create_training_graph에 대한 매개 변수, 기본 그래프
    # 모델 생성
    y_conv = build_model(is_training=True)    # 이전 모델 정의에서 학습을 작성할 때 드롭 아웃을 사용하므로 is_training은 True로 설정.
    # 손실 함수
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    print('cost:', cross_entropy)
    if is_quantify:
        # create_training_graph 함수를 추가하고 최적화하기 전에 손실함수를 먼저 실행해야함.
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=0)
    # 최적화
    optimize = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 정확도
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    # 인식 정확도 제공 [bool 값 목록을 반환합니다. 어떤 부분이 올바른지 확인하려면 해당 부분을 부동 소수점 값으로 변환 한 다음 평균을 표시해야함.
    # 예를 들어 [True, False, True, True]는 [1,0,1,1]로 변환되므로 정확도는 0.75입니다.]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 훈련에 필요한 데이터를 반환
    return dict(
        x=x,
        y=y_,
        keep_prob=keep_prob,
        optimize=optimize,
        cost=cross_entropy,
        correct_prediction=correct_prediction,
        accuracy=accuracy,
    )

def reshape_batch(batch):
    rebatch = []
    for item in batch:
        b = item.reshape(28,28)
        rebatch.append(b)
    return rebatch

# train 시작
def train_network(graph,ckpt,point_dir,pbtxt):
    # 초기화
    init = tf.global_variables_initializer()
    # 모델 정보 저장을 위해 Saver 호출.
    saver = tf.train.Saver()
    # 컨텍스트를 만들고 훈련 시작 sess.run (init)
    with tf.Session() as sess:
        sess.run(init)
        # 총 20,000 회 훈련, 정확도는 99 % 이상
        for i in range(20000):
        # 한 번에 50 개의 이미지 처리
            batch = mnist.train.next_batch(50)
            # 100 회마다 저장 및 출력
            if i % 100 == 0:
            # feed_dict가 데이터를 공급하면 데이터가 28x28로 완전히 재구성됨.
                train_accuracy = sess.run([graph['accuracy']], feed_dict={
                                                                           graph['x']:reshape_batch(batch[0]),    # 배치 [0] 저장된 이미지 데이터
                                                                           graph['y']:batch[1],    # 배치 [1] 저장된 태그
                                                                           graph['keep_prob']: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy[0]))
            sess.run([graph['optimize']], feed_dict={
                                                       graph['x']:reshape_batch(batch[0]),
                                                       graph['y']:batch[1],
                                                       graph['keep_prob']:0.5})
        test_accuracy = sess.run([graph['accuracy']], feed_dict={
                                                                  graph['x']: reshape_batch(mnist.test.images),
                                                                  graph['y']: mnist.test.labels,
                                                                  graph['keep_prob']: 1.0})
        print("Test accuracy %g" % test_accuracy[0])
        # ckpt (체크 포인트) 및 pbtxt를 저장하는 단계.
        saver.save(sess, ckpt)
        tf.train.write_graph(sess.graph_def,point_dir,pbtxt, True)
        print(tf.trainable_variables())
        print(tf.get_variable('W_fc2',[1024, 10]).value)


if __name__ == "__main__":
    ckpt = './checkpoint/mnist.ckpt'
    point_dir = './checkpoint'
    pbtxt = 'mnist.pbtxt'
    g1 = create_training_graph(False)
    train_network(g1,ckpt,point_dir,pbtxt)

