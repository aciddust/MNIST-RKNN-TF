import numpy as np
from rknn.api import RKNN
import datetime

if __name__ == '__main__':
    rknn = RKNN(verbose=False)
    rknn.load_rknn('./rknn_output/rknn_linear_regression.rknn')
    rknn.init_runtime()
    print('init runtime done')
    
    prev = datetime.datetime.now()
    input_data = np.array([3,20,4,8], dtype='float32')
    output = rknn.inference(inputs=input_data)
    print(datetime.datetime.now()-prev)
    print(output)
    rknn.release()

print('Done')
