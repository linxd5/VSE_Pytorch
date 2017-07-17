import numpy
from torch.autograd import Variable


def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)


def xavier_weight(tensor):
    if isinstance(tensor, Variable):
        xavier_weight(tensor.data)
        return tensor

    nin, nout = tensor.size()[0], tensor.size()[1]
    r = numpy.sqrt(6.) / numpy.sqrt(nin + nout)
    return tensor.normal_(0, r)

