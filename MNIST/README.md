# MNIST-RKNN-TF

MNIST for RKNN (with TF)



## ENV

Python 3.5.6 ( * I'm using Conda Environment )

Tensorflow 1.11.0

numpy 1.18.0 (* maybe u need downgrade version.)



## USAGE

```bash
python get_image.py
cd ../MNIST_data && gzip -d t10k-images.idx3-ubyte.gz
cd ../MNIST
python train.py
python freeze.py
python rknn_transfer.py
python tf_predict.py
python rk_predict.py
```

In order from the top.



