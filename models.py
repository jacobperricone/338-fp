import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import torch.optim as optim
import torch.nn.init as init
from utils import *


class ConvDQNSoftmax(nn.Module):
    def __init__(self, output_size):
        super(ConvDQNSoftmax, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.head = nn.Linear(256, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc(out.view(out.size(0), -1)))
        out = self.softmax(self.head(out))
        return out


class ConvDQNRegressor(nn.Module):
    def __init__(self):
        super(ConvDQNRegressor, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc(out.view(out.size(0), -1)))
        out = self.head(out)
        return out


class DQNSoftmax(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNSoftmax, self).__init__()

        self.observation_size = input_size
        self.action_size = output_size
        self.hidden_size = 64
        self.h1 = nn.Linear(self.observation_size, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h3 = nn.Linear(self.hidden_size, self.action_size)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.initialize_weights()

    def initialize_weights(self):
        init.xavier_uniform_(self.h1.weight)
        init.xavier_uniform_(self.h2.weight)
        init.xavier_uniform(self.h3.weight)

    def forward(self, x):
        out = self.activation(self.h1(x))
        out = self.activation(self.h2(out))
        out = self.softmax(self.h3(out))
        return out


class DQNRegressor(nn.Module):
    def __init__(self, input_size):
        super(DQNRegressor, self).__init__()

        self.input_size = input_size
        self.hidden_size = 64
        self.h1 = nn.Linear(self.input_size, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h3 = nn.Linear(self.hidden_size, 1)
        self.activation = nn.Tanh()

    def initialize_weights(self):
        init.xavier_uniform(self.h1.weight)
        init.xavier_uniform(self.h2.weight)
        init.xavier_uniform(self.h3.weight)

    def forward(self, x):
        out = self.activation(self.h1(x))
        out = self.activation(self.h2(out))
        out = self.h3(out)
        return out

#
class ValueFunctionWrapper(nn.Module):
    def __init__(self, model, lr):
        super(ValueFunctionWrapper, self).__init__()

        self.model = model
        self.loss_fn = nn.MSELoss(size_average=False)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.mix_frac = .1

    def predict(self, observations):
        return self.model.forward(observations)

    def fit(self, observations, labels):
        for i in range(0, 10):

            predicted = self.predict(observations)
            loss = self.loss_fn(predicted, labels*self.mix_frac + (1 - self.mix_frac)*predicted.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predicted = self.predict(observations)

        # check
        current_params = parameters_to_vector(self.model.parameters())
        if any(np.isnan(current_params.data.cpu().numpy())):
            print("*** Adam optimization diverged. Exiting!")
            exit()
        return loss


# class ValueFunctionWrapper(nn.Module):
#     def __init__(self, model, lr):
#         super(ValueFunctionWrapper, self).__init__()
#         self.model = model
#         self.loss_fn = nn.MSELoss(size_average=False)
#         self.optimizer = optim.LBFGS(self.model.parameters(), lr=lr)
#         self.mix_frac = .1
#
#     def predict(self, observations):
#         return self.model.forward(observations)
#
#     def fit(self, observations, labels):
#         def closure():
#             predicted = self.predict(observations)
#             loss = torch.pow(predicted - labels, 2).sum()
#             self.optimizer.zero_grad()
#             loss.backward()
#             return loss
#
#         self.optimizer.step(closure)
#
#         # check
#         current_params = parameters_to_vector(self.model.parameters())
#         if any(np.isnan(current_params.data.cpu().numpy())):
#             print("*** Adam optimization diverged. Exiting!")
#             exit()
#         return
