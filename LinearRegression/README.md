# Linear Regression

WX + b



## ENV

Python 3.5.6 ( * I'm using Conda Environment)

Tensorflow 1.11.0

Numpy 1.18.0  



## USAGE

```bash
python run.py
python freeze_graph.py \
--input_binary=true \
--input_graph=./output/linear_regression.pb \
--input_checkpoint=./output/linear_regression.ckpt \
--output_graph=./output/frozen_linear_regression.pb \
--output_node_names=hypothesis
python rknn_transfer.py
python rknn_predict.py
```







