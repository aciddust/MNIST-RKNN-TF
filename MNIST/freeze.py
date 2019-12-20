'''
ckpt에서 그래프/가중치 정보를 읽고 pb 파일로 저장 한 후 이전 단계와 같이 is_quantify를 False로 설정.
'''

import tensorflow as tf
import os.path
from model import build_model

# inference 그래프 생성
def create_inference_graph():
    """Build the mnist model for evaluation."""
# 출력 생성
    logits = build_model(is_training=False)
    return logits
    # # 분류 결과 획득
    # tf.nn.softmax(logits, name='output')
def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    saver.restore(sess, start_checkpoint)

def frozen(is_quantify,ckpt,pbtxt):
    # 모델을 생성하고 생성된 모델의 가중치를 불러옴
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
    # 추론 그래프
        logits = create_inference_graph()  
    # create_eval_graph ()를 추가하여 tflite에 적합한 형식으로 변환. 다음 명령문에 경로가 있으면 경로를 변경해야합니다.
        if is_quantify:
            tf.contrib.quantize.create_eval_graph()
        load_variables_from_checkpoint(sess, ckpt)
        # Turn all the variables into inline constants inside the graph and save it.
    # frozen：ckpt + pbtxt
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['y_conv'])
    # 최종 PB 모델 저장
        tf.io.write_graph(
            frozen_graph_def,
            os.path.dirname(pbtxt),
            os.path.basename(pbtxt),
            as_text=False)
        tf.compat.v1.logging.info('Saved frozen graph to %s', pbtxt)

if __name__ == "__main__":
    ckpt = './checkpoint/mnist.ckpt'
    pbtxt = 'mnist_frozen_graph.pb'
    frozen(False,ckpt,pbtxt)
    #is_quantify False   mnist_frozen_graph_not_28x28.pb
    # ckpt = './checkpoint_not/mnist.ckpt'
    # pbtxt = 'mnist_frozen_graph_not.pb'
    # frozen(False,ckpt,pbtxt)
    # ckpt = './test/mnist.ckpt'
    # pbtxt = 'test.pb'
    # frozen(False,ckpt,pbtxt)
