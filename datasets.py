"""
Dataset loading
"""
import numpy

path_to_data = 'data/'

def load_dataset(name='f8k', load_test=False):
    """
    Load captions and image features
    """
    loc = path_to_data + name + '/'

    if load_test:
        # Captions
        test_caps = []
        with open(loc+name+'_test_caps.txt', 'rb') as f:
            for line in f:
                test_caps.append(line.strip())
        # Image features
        test_ims = numpy.load(loc+name+'_test_ims.npy')
        return (test_caps, test_ims)
    else:
        # Captions
        train_caps, dev_caps = [], []
        with open(loc+name+'_train_caps.txt', 'rb') as f:
            for line in f:
                train_caps.append(line.strip())

        with open(loc+name+'_dev_caps.txt', 'rb') as f:
            for line in f:
                dev_caps.append(line.strip())

        # Image features
        train_ims = numpy.load(loc+name+'_train_ims.npy')
        dev_ims = numpy.load(loc+name+'_dev_ims.npy')

        return (train_caps, train_ims), (dev_caps, dev_ims)