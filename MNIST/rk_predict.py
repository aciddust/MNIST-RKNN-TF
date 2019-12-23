import numpy as np
from PIL import Image
from rknn.api import RKNN
import cv2
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
print(mnist.test.images[0].shape)
# 모델의 출력을 분석하여 확률이 가장 높고 해당 확률이 높은 제스처 획득.
def get_predict(probability):
    data = probability[0][0]
    data = data.tolist()
    max_prob = max(data)
    return data.index(max_prob), max_prob
# return data.index(max_prob), max_prob;
def load_model(model_name):
    # RKNN 객체 생성
    rknn = RKNN()
    # RKNN 모델 불러 오기
    print('-->loading model')
    rknn.load_rknn(model_name)
    print('loading model done')
    # RKNN 런타임 환경 초기화
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
       print('Init runtime environment failed')
       exit(ret)
    print('done')
    return rknn
def predict(rknn,length):
    '''
    acc_count = 0
    for i in range(length):
        # im = mnist.test.images[i]
        im = Image.open("../MNIST_data/mnist_test/test_%d.jpg"%i)   # 사진로드
        im = im.resize((28,28),Image.ANTIALIAS)
        im = np.asarray(im)
        outputs = rknn.inference(inputs=[im])
        pred, prob = get_predict(outputs)
        if i ==0:
            print(outputs)
            print(prob)
            print(pred)
        if i ==100 or i ==500 or i ==1000 or i ==10000:
            result = float(acc_count)/i
            print('result%d:'%i,result)
        if list(mnist.test.labels[i]).index(1) == pred:
            acc_count += 1
    result = float(acc_count)/length
    print('result:',result)
    '''

    acc_count = 0
    length = len(mnist.test.images)
    for i in range(length):
        im = mnist.test.images[i]
        outputs = rknn.inference(inputs=[im])   # 추론 실행 및 추론 결과 얻기
        pred, prob = get_predict(outputs)     # 추론 결과를 시각적 정보로 전환
        if list(mnist.test.labels[i]).index(1) == pred:
            acc_count += 1
        if i%100 == 0:
            print('입력 값 : {0} / 예상한 값 : {1} / 예측 정확도 : {2} / 맞은 갯수 : {3}'.format(list(mnist.test.labels[i]).index(1),pred, prob, acc_count))
    result = float(acc_count)/length
    print('result:',result)
if __name__=="__main__":
    # 해당 양자화 또는 비 양자화 rknn 모델로 변경
    model_name = './mnist.rknn'
    length = 10000
    rknn = load_model(model_name)
    predict(rknn,length)

    rknn.release()