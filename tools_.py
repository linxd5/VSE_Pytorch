
# coding: utf-8

import numpy
from collections import defaultdict
import torch
from torch.autograd import Variable

def encode_sentences(model, X, verbose=False, batch_size=128):
    """
    Encode sentences into the joint embedding space
    """
    features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i,s in enumerate(captions):
        ds[len(s)].append(i)


    # quick check if a word is in the dictionary
    d = defaultdict(lambda : 0)
    for w in model['worddict'].keys():
        d[w] = 1

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]
            caption = [captions[c] for c in caps]

            seqs = []
            for i, cc in enumerate(caption):
                seqs.append([model['worddict'][w] if d[w] > 0 and model['worddict'][w] < model['options']['n_words'] else 1 for w in cc])
            x = numpy.zeros((k+1, len(caption))).astype('int64')
            for idx, s in enumerate(seqs):
                x[:k,idx] = s

            x = Variable(torch.from_numpy(x).cuda())
            ff = model['img_sen_model'].forward_sens(x)

            for ind, c in enumerate(caps):
                features[c] = ff[ind].data.cpu().numpy()

    features = Variable(torch.from_numpy(features).cuda())
    return features



def encode_images(model, IM):
    """
    Encode images into the joint embedding space
    """
    IM = Variable(torch.from_numpy(IM).cuda())
    images = model['img_sen_model'].forward_imgs(IM)
    return images

