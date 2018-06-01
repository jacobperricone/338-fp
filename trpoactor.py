import numpy as np
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from models import ValueFunctionWrapper
from utils import *
import copy 
import random 
import sys

eps = 1e-8

class TRPOActor:
    def __init__(self, args, env, policy_model, value_function_model):
        self.args = args
        self.env = env
        self.policy_model = policy_model
        self.value_function_model = ValueFunctionWrapper(value_function_model, self.args.value_function_lr)
        self.column_headings = {'features': 0,
                                'actions': 1,
                                'returns': 2,
                                'advantages': 3}

        self.ob_filter, self.rew_filter = ZFilter(env.observation_space.shape, clip=5), ZFilter((), demean=False,clip = 10)


    def set_policy_weights(self, theta):

        old_model = copy.deepcopy(self.policy_model)
        old_model.load_state_dict(self.policy_model.state_dict())
        vector_to_parameters(theta, self.policy_model.parameters())
        assert(self.policy_model.parameters()!= old_model.parameters())

    def set_value_function_weights(self, theta):
        old_model = copy.deepcopy(self.value_function_model)
        old_model.load_state_dict(self.value_function_model.state_dict())
        vector_to_parameters(theta, self.value_function_model.parameters())
        assert(self.value_function_model.parameters()!= old_model.parameters())

    def sample_action_from_policy(self, observation):
        observation_tensor = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0)
        action_distribution = self.policy_model(Variable(observation_tensor, requires_grad=True))
        m = torch.distributions.Categorical(action_distribution)
        action = m.sample().unsqueeze(0)
        return action, action_distribution,m

    def sample_episode(self, num_timesteps=sys.maxsize):
        observations, actions, rewards, action_distributions = [], [], [], []
        observation = self.env.reset()
        for i in range(self.args.max_pathlength - 1):
            observation = self.ob_filter(observation)
            observations.append(observation)
            action, action_distribution, m = self.sample_action_from_policy(observation)
            # print(action.data[0,0])
            # print(action, action_distribution)
            # print(action, action_distribution)
            actions.append(action)
            action_distributions.append(action_distribution)
            # entropy += -(action_dist * action_dist.log()).sum()
            observation, reward, done, info = self.env.step(action.data[0,0].item())
            rewards.append(reward)
            reward = self.rew_filter(reward)

            if done or i == self.args.max_pathlength - 2 or i == num_timesteps - 1:
                observations = np.concatenate(np.expand_dims(observations, 0))
                times = np.arange(observations.shape[0]).reshape(-1, 1) / self.args.max_pathlength
                features = torch.from_numpy(np.hstack((observations, times)).astype(np.float32))  # rather than constructing using Tensor, use from_numpy()
                rewards = np.array(rewards)
                returns = torch.from_numpy(np.expand_dims(discount(rewards, self.args.gamma), -1).astype(np.float32).copy())  # copy to resolve negative stride issue
                baseline = self.value_function_model.predict(Variable(features)).data.numpy()
                baseline = np.append(baseline, 0 if done else baseline[-1])
                delta = rewards + self.args.gamma * baseline[1:] - baseline[:-1]
                advantage = torch.from_numpy(np.expand_dims(discount(delta, self.args.gamma * self.args.lam), -1).astype(np.float32).copy())
                path = [features, torch.cat([item for item in actions]), returns, advantage]
                # delta = discount(rewards, self.args.gamma*self.args.lam)  - baseline[:-1]
                # advantage = torch.from_numpy(np.expand_dims(delta.astype(np.float32).copy(),-1))
                # path = [features, torch.cat([item for item in actions]), returns, advantage]
                return path, torch.cat([item for item in action_distributions]), rewards.sum()

    # Need to calculate policy gradient here, because we need to use the forward passes through the policy
    # network in order to compute policy gradients (forward passes only collected in rollouts).  In distributed memory,
    # learner has no access to the policy network used by actors in rollout so will not have access to gradients
    def calc_policy_grads(self, paths, action_distributions):
        advantages = paths[self.column_headings['advantages']]
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)  # first, standardize advantages
        actions = paths[self.column_headings['actions']]
        new_p = torch.cat(action_distributions).gather(1, actions)
        old_p = new_p.detach()
        prob_ratio = new_p / old_p
        surrogate_loss = -torch.mean(prob_ratio * Variable(advantages))

        # derivative of surrogate loss WRT model parameters
        self.policy_model.zero_grad()
        surrogate_loss.backward(retain_graph = True)  # probably need retain graph for second and future iterations of learning, and need to zero out grads before computing grads
        policy_gradient = parameters_to_vector([v.grad for v in self.policy_model.parameters()]).squeeze(0)
        print(torch.abs(policy_gradient).sum())
        return policy_gradient.data

    def rollouts(self, num_timesteps, seed_iter):
        paths, action_distributions = [], []
        steps_episodes_rewards = np.zeros(3, dtype=np.int)
        if self.args.rollout_limit == "timesteps":  # for equal timestep rollouts
            while steps_episodes_rewards[0] < num_timesteps:
                np.random.seed(next(seed_iter))
                path, action_dist, reward = self.sample_episode(num_timesteps - steps_episodes_rewards[0])
                steps_episodes_rewards[0] += path[0].size()[0]
                paths.append(path)
                action_distributions.append(action_dist)
                if (steps_episodes_rewards[0] <= num_timesteps):  # only record full episodes for averaging!
                    steps_episodes_rewards[1] += 1
                    steps_episodes_rewards[2] += reward
        elif self.args.rollout_limit == "episodes":  # for equal number of episode rollouts
            while steps_episodes_rewards[0] < num_timesteps:
                np.random.seed(next(seed_iter))
                path, action_dist, reward = self.sample_episode()
                steps_episodes_rewards[0] += path[0].size()[0]
                paths.append(path)
                action_distributions.append(action_dist)
                steps_episodes_rewards[1] += 1
                steps_episodes_rewards[2] += reward
        else:
            print("*** Problem in rollout(): invalid collection limiting strategy")
            exit()
        paths_tensor = [torch.cat([path[i] for path in paths]) for i in range(len(paths[0]))]
        grads_tensor = self.calc_policy_grads(paths_tensor, action_distributions)
        paths_tensor[self.column_headings['actions']] = paths_tensor[self.column_headings['actions']].data  # convert variable to tensor
        return paths_tensor, grads_tensor, steps_episodes_rewards
