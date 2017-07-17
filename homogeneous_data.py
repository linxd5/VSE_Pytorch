import numpy
import copy
import sys


class HomogeneousData():

    def __init__(self, data, batch_size=128, maxlen=None):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0]
        self.feats = self.data[1]

        # find the unique lengths
        self.lengths = [len(cc.split()) for cc in self.caps]
        self.len_unique = numpy.unique(self.lengths)

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

        # current counter
        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

        caps = [self.caps[ii] for ii in curr_indices]
        feats = [self.feats[ii] for ii in curr_indices]

        return caps, feats

    def __iter__(self):
        return self


def prepare_data(caps, features, worddict, maxlen=None, n_words=10000):
    """
    Put data into format useable by the model
    """
    seqs = []
    feat_list = []
    for i, cc in enumerate(caps):
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])
        feat_list.append(features[i])

    lengths = [len(s) for s in seqs]

    y = numpy.asarray(feat_list, dtype=numpy.float32)

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s

    return x, y
