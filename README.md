# Visual-semantic-embedding

Pytorch Code for the image-sentence ranking methods from *[Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](https://arxiv.org/abs/1411.2539)* (Kiros,Salakhutdinov, Zemel, 2014).

Images and sentences are mapped into a common vector space, where the sentence representation is computed using LSTM. This project contains training code and pre-trained models for Flick8K,flickr30K and MSCOCO.

1. **Thanks to ryankiros's implementation of the model in Theano: https://github.com/ryankiros/visual-semantic-embedding**

# Results

Below is a table of results obtained using the code from this repository, comparing the numbers reported in paper. aR@K is the Recall@K for image annotation (higher is better), while sR@K is the Recall@K for image search (higher is better). Medr is the median rank of the closest ground truth (lower is better).

All experiments below train on GTX TITAN (1 card).

## Flickr8K

learning_rate: 0.001, batch_size: 128, validFreq: 100, dim&dim_word: 1000


| Method       | aR@1 | aR@5 | aR@10 | aMedr | sR@1 | sR@5 | sR@10 | sMedr |
| ------------ | ---- | ---- | ----- | ----- | ---- | ---- | ----- | ----- |
| Paper        | 18.0 | 40.9 | 55.0  |   8   | 12.5 | 37.0 | 51.5  |   10  |
| This project | 23.9 | 49.1 | 61.3  |   6   | 16.9 | 41.8 | 54.4  |   9  |

## Flickr30K
learning_rate: 0.01, batch_size: 200, validFreq: 100, dim&dim_word: 1000


| Method       | aR@1 | aR@5 | aR@10 | aMedr | sR@1 | sR@5 | sR@10 | sMedr |
| ------------ | ---- | ---- | ----- | ----- | ---- | ---- | ----- | ----- |
| Paper        | 23.0 | 50.7 | 62.9  |   5   | 16.8 | 42.0 | 56.5  |   8   |
| This project | 29.0 | 57.7 | 67.3  |   4   | 21.5 | 48.0 | 59.0  |   6  |

Cost 2G GPU memory and 485s.

## MSCOCO
learning_rate: 0.01, batch_size: 300, validFreq: 100, dim&dim_word: 1000


| Method       | aR@1 | aR@5 | aR@10 | aMedr | sR@1 | sR@5 | sR@10 | sMedr |
| ------------ | ---- | ---- | ----- | ----- | ---- | ---- | ----- | ----- |
| This project | 38.1 | 72.7 | 84.7  |   2   | 31.7 | 67.9 | 81.2  |   3  |

Cost 2G GPU memory and 1180s.





# Dependencies

This code is written in python, To use it you will need:
```Python
$ virtualenv env
$ source env/bin/activate
$ pip install torch-0.1.11.post5-cp27-none-linux_x86_64.whl
$ pip install torchvision
```

# Getting started

You will first need to download the dataset files and pre-trained models. These can be obtained by runing:
```shell
wget http://www.cs.toronto.edu/~rkiros/datasets/f8k.zip
wget http://www.cs.toronto.edu/~rkiros/datasets/f30k.zip
wget http://www.cs.toronto.edu/~rkiros/datasets/coco.zip
wget http://www.cs.toronto.edu/~rkiros/models/vse.zip
```

Each of the dataset files contains the captions as well as [VGG features](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) from the 19-layer model. Flickr8K comes with a pre-defined train/dev/test split, while for Flickr30K and MS COCO we use the splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). Note that the original images are not included with the dataset.


# Training new models

Open `test.py` and specify the hyperparameters that you would like. Below we describe each of them in detail:
- data: The dataset to train on (f8k, f30k or coco).
- dim_image: The dimensionality of the image features. This will be 4096 for VGG.
- margin: The margin used for computing the pairwise ranking loss. Should be between 0 and 1.
- max_epochs: The number of epochs used for training.
- batch_size: The size of a minibatch.
- dim: The dimensionality of the learned embedding space (also the size of the RNN state).
- dim_word: The dimensionality of the learned word embeddings.
- maxlen_w: Sentences longer then this value will be ignored.
- dispFreq: How often to display training progress.
- validFreq: How often to evaluate on the development set.
- saveto: The location to save the model.

As the model trains, it will periodically evaluate on the development set (validFreq) and re-save the model each time performance on the development set increases. Generally you shouldn't need more than 15-20 epochs of training on any of the datasets. Once the models are saved, you can load and evaluate them in the same way as the pre-trained models.



# Others

This project is a simplified version of my another project -- [ImageTextRetrieval](https://github.com/linxd5/ImageTextRetrieval/blob/master/README.md), which is used to retrieval **architecture** images and text.
