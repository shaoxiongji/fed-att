# Attentive Federated Learning

This repository contains the code for the paper [Learning Private Neural Language Modeling with Attentive Aggregation](https://arxiv.org/abs/1812.07108), which is an attentive extention of federated aggregation. A brief introductionary blog is avaiable [here](https://shaoxiongji.github.io/2019/07/11/attentive-federated-learning.html).

Further reference: a universal federated learning repository implemented by PyTorch - [Federated Learning - PyTorch](https://github.com/shaoxiongji/federated-learning).  

## Run
Refer to the ```README.md``` under the data folder and download the datasets into their corresponding folders. Enter the source code folder to run the scripts with arguments assigned using ```argparse``` package.
```
cd src
python run.py
```

See configs in ```src/utils/options.py```

## Requirements
Python 3.6  
PyTorch 0.4.1 

## Cite
```
@inproceedings{ji2019learning,
  title={Learning Private Neural Language Modeling with Attentive Aggregation},
  author={Ji, Shaoxiong and Pan, Shirui and Long, Guodong and Li, Xue and Jiang, Jing and Huang, Zi},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2019}
}
```
