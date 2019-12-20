import cv2
import numpy as np
from rknn.api import RKNN

if __name__ == '__main__':
    rknn = RKNN(verbose=False)
    rknn.load_rknn('wxb.rknn')
    rknn.init_runtime()
    print("init runtime done")
    
    input_data = np.array([10.,], dtype='float32')
    output = rknn.inference(inputs=input_data)
    
    print(output)
    rknn.release()

print('Done')