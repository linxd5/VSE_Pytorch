# coding: utf-8
import torch
from utils import l2norm, xavier_weight
from torch.autograd import Variable


class ImgSenRanking(torch.nn.Module):
    def __init__(self, model_options):
        super(ImgSenRanking, self).__init__()
        self.linear = torch.nn.Linear(model_options['dim_image'], model_options['dim'])
        self.lstm = torch.nn.LSTM(model_options['dim_word'], model_options['dim'], 1)
        self.embedding = torch.nn.Embedding(model_options['n_words'], model_options['dim_word'])
        self.model_options = model_options
        self.init_weights()

    def init_weights(self):
        xavier_weight(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, x, im):
        x_emb = self.embedding(x)
        im = self.linear(im)

        _, (x_emb, _) = self.lstm(x_emb)
        x_emb = x_emb.squeeze(0)

        return l2norm(x_emb), l2norm(im)

    def forward_sens(self, x):
        x_emb = self.embedding(x)

        _, (x_emb, _) = self.lstm(x_emb)
        x_cat = x_emb.squeeze(0)
        return l2norm(x_cat)

    def forward_imgs(self, im):
        im = self.linear(im)
        return l2norm(im)

class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):
        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

        for i in xrange(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return cost_s.sum() + cost_im.sum()
