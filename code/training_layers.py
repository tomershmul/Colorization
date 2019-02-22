from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import os
import sklearn.neighbors as nn



class PriorFactor():
    ''' Class handles prior factor '''

    def __init__(self, alpha, gamma=0, verbose=True, priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.
        self.uni_probs = self.uni_probs / np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix ** -self.alpha
        self.prior_factor = self.prior_factor / np.sum(self.prior_probs * self.prior_factor)  # re-normalize

    def forward(self, data_ab_quant, axis=1):
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        return corr_factor[:, na(), :]

class NNEncLayer(object):
    ''' Layer which encodes ab map into Q colors
    OUTPUTS
        top[0].data     NxQ
    '''

    def __init__(self):
        self.NN = 3
        self.sigma = 5
        self.ENC_DIR = './resources/'
        self.nnenc = NNEncode(self.NN, self.sigma, km_filepath=os.path.join(self.ENC_DIR, 'pts_in_hull.npy'))

    def forward(self, x):
        encode=self.nnenc.encode_points_mtx_nd(x)
        max_encode=np.argmax(encode,axis=1).astype(np.int32)
        return encode,max_encode


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''

    def __init__(self, NN, sigma, km_filepath):
        self.cc = np.load(km_filepath)
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        P = pts_flt.shape[0]
        if (sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P, self.K))
            self.p_inds = np.arange(0, P, dtype='int')[:, na()]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]
        self.pts_enc_flt[self.p_inds, inds] = wts

        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)
        return pts_enc_nd

# *****************************
# ***** Utility functions *****
# *****************************


def na():  # shorthand for new axis
    return np.newaxis


def flatten_nd_array(pts_nd, axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.dim()
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = tuple(np.concatenate((nax, np.array(axis).flatten()), axis=0).tolist())
    pts_flt = pts_nd.permute(axorder)
    pts_flt = pts_flt.contiguous().view(NPTS.item(), SHP[axis].item())
    return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.dim()
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices

    if (squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        NEW_SHP = SHP[nax].tolist()

        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = tuple(np.argsort(axorder).tolist())
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    return pts_out


def non_gray_mask(self, bottom):
    ''' Layer outputs a mask based on if the image is grayscale or not '''
    bottom=bottom.numpy()
    # if an image has any (a,b) value which exceeds threshold, output 1
    return (np.sum(np.sum(np.sum((np.abs(bottom) > 5).astype('float'), axis=1), axis=1), axis=1) > 0)[:,
                       na(), na(), na()].astype('float')

