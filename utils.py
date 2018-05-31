import numpy as np
import torch
import scipy.signal as signal
# import torch.distributed as dist
from mpi4py import MPI

##############
# MATH UTILS #
##############

def discount(x, gamma):
	"""
	Compute discounted sum of future values
	out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
	"""
	return signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred, y):
	"""
	Var[ypred - y] / var[y].
	https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
	"""
	assert y.ndim == 1 and ypred.ndim == 1
	vary = np.var(y)
	return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

###############
# TORCH UTILS #
###############

use_cuda = torch.cuda.is_available()

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
				path = np.empty([size * data_paths[i].size()[0], data_paths[i].size()[1]], dtype=data_paths[i].numpy().dtype)
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
				sendcounts = sendcounts0*data_paths[i].size()[1]
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
		episodes_rewards = np.empty(2, dtype = np.int)
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
