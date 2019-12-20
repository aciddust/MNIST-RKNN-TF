'''
양자화된 rknn 모델에는 해당 픽처 세트가 필요하므로
먼저 mnist 데이터 디렉토리에 해당 데이터 세트를 가져 와서
t10k-images-idx3-ubyte.gz를 압축 해제 한 다음
get_image.py를 실행하여 원래 압축 된 데이터를 픽처로 변환해야함.

동시에, 양자화에 필요한 dataset.txt 파일을 얻으십시오.
'''

import struct
import numpy as np
#import matplotlib.pyplot as plt
import PIL.Image
from PIL import Image
import os

os.system("mkdir ../MNIST_data/mnist_test")
filename='../MNIST_data/t10k-images.idx3-ubyte'
dataset = './dataset.txt'
binfile=open(filename,'rb')
buf=binfile.read()
index=0
data_list = []
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')
for image in range(0,numImages):
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')
    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
    im=Image.fromarray(im)
    im.save('../MNIST_data/mnist_test/test_%s.jpg'%image,'jpeg')
    data_list.append('../MNIST_data/mnist_test/test_%s.jpg\n'%image)
with open(dataset,'w+') as ff:
    ff.writelines(data_list)