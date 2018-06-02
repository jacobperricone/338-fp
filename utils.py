import numpy as np
import torch
import scipy.signal as signal
# import torch.distributed as dist
from mpi4py import MPI


##############
# MATH UTILS #
##############
import os
import shutil

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def explained_variance_1d(ypred, y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


###############
# TORCH UTILS #
###############

use_cuda = torch.cuda.is_available()




def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def Variable(tensor, *args, **kwargs):
    if use_cuda:
        return torch.autograd.Variable(tensor, *args, **kwargs).cuda()
    else:
        return torch.autograd.Variable(tensor, *args, **kwargs)


def Tensor(nparray):
    dtype = nparray.dtype
    if use_cuda:
        return torch.Tensor(nparray).cuda()
    else:
        return torch.Tensor(nparray)


class RunningStat:
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


#############
# MPI UTILS #
#############




def gather_paths(data_paths, data_grads, data_steps, balancing):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    paths = []
    if balancing == "timesteps":  # gather if we have equal time steps per process
        for i in range(len(data_paths)):
            if rank == 0:
                path = np.empty([size * data_paths[i].size()[0], data_paths[i].size()[1]],
                                dtype=data_paths[i].numpy().dtype)
            else:
                path = None
            comm.Gather(data_paths[i].numpy(), path, root=0)
            if rank == 0:
                paths.append(torch.from_numpy(path))
    elif balancing == "episodes":  # gather if we have equal episodes per process
        sendcounts0 = np.array(comm.gather(data_steps[0], root=0))
        if rank == 0:
            totalsteps = np.sum(sendcounts0)
        # print(totalsteps)
        for i in range(len(data_paths)):
            if rank == 0:
                path = np.empty([totalsteps, data_paths[i].size()[1]], dtype=data_paths[i].numpy().dtype)
                sendcounts = sendcounts0 * data_paths[i].size()[1]
            else:
                path = None
                sendcounts = None
            comm.Gatherv(data_paths[i].numpy(), [path, sendcounts], root=0)
            if rank == 0:
                paths.append(torch.from_numpy(path))
    else:
        print("*** Problem in gather_paths(): invalid parallel balancing strategy")
        exit()

    # sum collected rewards
    if rank == 0:
        episodes_rewards = np.empty(2, dtype=np.int)
    else:
        episodes_rewards = None
    comm.Reduce(data_steps[1:], episodes_rewards, op=MPI.SUM, root=0)

    # mean policy gradients
    if rank == 0:
        gradients = np.empty(data_grads.size(), dtype=data_grads.numpy().dtype)
    else:
        gradients = None

    comm.Reduce(data_grads.numpy(), gradients, op=MPI.SUM, root=0)
    if rank == 0:
        gradients = torch.from_numpy(gradients / size)
    return paths, gradients, episodes_rewards
