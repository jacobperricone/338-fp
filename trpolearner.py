import numpy as np
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from models import ValueFunctionWrapper
from utils import *
import copy

eps = 1e-8

class TRPOLearner:
    def __init__(self, args, env, policy_net, value_net, orderings):
        self.args = args
        self.env = env
        self.policy_net = policy_net
        self.value_net = ValueFunctionWrapper(value_net, self.args.value_function_lr)
        self.orderings = orderings

    def get_policy_weights(self):
        return parameters_to_vector(self.policy_net.parameters())

    def get_value_function_weights(self):
        return parameters_to_vector(self.value_net.parameters())

    def mean_kl_divergence(self, model):
        """
        Returns an estimate of the average KL divergence between a given model and self.policy_model (-> variable)
        """
        actprob = model(self.observations_tensor).detach() + 1e-8
        old_actprob = self.policy_net(self.observations_tensor)
        return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()

    def hessian_vector_product(self, vector):
        """
        Returns the product of the Hessian of the KL divergence and the given vector (-> tensor)
        """
        self.policy_net.zero_grad()
        mean_kl_div = self.mean_kl_divergence(self.policy_net)
        kl_grad = torch.autograd.grad(mean_kl_div, self.policy_net.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        grad_vector_product = torch.sum(kl_grad_vector * Variable(vector))
        grad_grad = torch.autograd.grad(grad_vector_product, self.policy_net.parameters())
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grad_grad])
        return fisher_vector_product.data + (self.args.cg_damping * vector)

    def conjugate_gradient(self, b, cg_iters=10, residual_tol=1.0e-10):
        """
        Returns F^(-1) * b where F is the Hessian of the KL divergence (-> tensor)
        """
        p = b.clone()
        r = b.clone()
        x = torch.zeros_like(b)
        rdotr = r.dot(r)
        for i in range(cg_iters):
            z = self.hessian_vector_product(p).squeeze(0)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p
            rdotr = newrdotr
            if rdotr < residual_tol:
                break
        return x

    def surrogate_loss(self, theta):
        """
        Returns the surrogate loss w.r.t. the given parameter vector theta (-> float)
        """
        old_theta = parameters_to_vector(self.policy_net.parameters())
        prob_old = self.policy_net(self.observations_tensor).gather(1, self.actions).data
        vector_to_parameters(theta, self.policy_net.parameters())
        prob_new = self.policy_net(self.observations_tensor).gather(1, self.actions).data
        vector_to_parameters(old_theta, self.policy_net.parameters())
        return -torch.mean((prob_new / (prob_old + eps)) * self.advantages)

    def linesearch(self, x, fullstep, expected_improve_rate):
        """
        Returns the parameter vector given by a linesearch (-> variable)
        """
        accept_ratio = .1
        max_backtracks = 10
        fval = self.surrogate_loss(x)
        # print("At start of linesearch, fval = {}".format(fval))
        for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            # print(("Search number {}...".format(_n_backtracks + 1)))
            xnew = Variable(x.data + stepfrac * fullstep)
            newfval = self.surrogate_loss(xnew)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio > accept_ratio and actual_improve > 0:
                # print("Final surrogate loss in line search = {}".format(fval))
                return xnew
        print("Problem: linesearch did not converge after {} iterations".format(max_backtracks))
        return x

    def learn(self, paths, policy_gradient, episodes_rewards, debug=1):
        self.observations_tensor = Variable(paths[self.orderings['features']][:, 0:self.env.observation_space.shape[0]], requires_grad=True)
        self.features_tensor = Variable(paths[self.orderings['features']])
        self.actions = Variable(paths[self.orderings['actions']])
        self.returns = Variable(paths[self.orderings['returns']])
        self.advantages = paths[self.orderings['advantages']]
        print(episodes_rewards)
        if episodes_rewards[0] == 0:
            episoderewards = 0
        else:
            episoderewards = episodes_rewards[1] / episodes_rewards[0]

        if not policy_gradient.nonzero().size()[0]:
            print("Policy gradient is 0. Skipping update ...")
            stats = {}
            stats["Avg_Reward"] = episoderewards
            stats["Timesteps"] = paths[0].size()[0]
            stats["Episodes"] = int(episodes_rewards[0])
            stats["Delta_KL"] = 0.0
            stats["Surrogate loss"] = -float('inf')
            return stats, self.get_policy_weights(), self.get_value_function_weights()

        # Use conjugate gradient algorithm to determine the step direction in theta space
        step_direction = self.conjugate_gradient(-policy_gradient)

        # Do line search to determine the stepsize of theta in the direction of step_direction
        shs = .5 * step_direction.dot(self.hessian_vector_product(step_direction))
        lm = np.sqrt(shs / self.args.max_kl)
        fullstep = step_direction / lm
        gdotstepdir = -(policy_gradient).dot(step_direction)
        theta = self.linesearch(parameters_to_vector(self.policy_net.parameters()), fullstep, gdotstepdir / lm)

        # Fit the estimated value function to the actual observed discounted rewards
        self.value_net.zero_grad()
        value_fn_params = parameters_to_vector(self.value_net.parameters())
        self.value_net.fit(self.features_tensor, self.returns)

        # Update parameters of policy model
        old_model = copy.deepcopy(self.policy_net)
        old_model.load_state_dict(self.policy_net.state_dict())
        if any(np.isnan(theta.data.numpy())):
            print("NaN detected. Skipping update...")
        else:
            vector_to_parameters(theta, self.policy_net.parameters())

        kl_after = self.mean_kl_divergence(old_model).data[0]
        surrogate_after = self.surrogate_loss(parameters_to_vector(self.policy_net.parameters()))

        stats = {}
        print(type(kl_after.data.numpy()))
        stats["Avg_Reward"] = episoderewards
        stats["Timesteps"] = paths[0].size()[0]
        stats["Episodes"] = int(episodes_rewards[0])
        tmp = float(kl_after.data.numpy())
        print(tmp)
        stats["Delta_KL"] = tmp
        stats["Surrogate loss"] = float(surrogate_after.data.numpy())
        print(stats)
        return stats, self.get_policy_weights(), self.get_value_function_weights()
