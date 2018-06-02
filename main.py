import numpy as np
import gym
from models import DQNRegressor, DQNSoftmax
from trpoactor import TRPOActor
from trpolearner import TRPOLearner
from utils import *
import argparse
import time
import logging
import shutil
import json
import itertools
import sys
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)

RESULTS_DIR = os.path.join(os.getcwd(), 'results')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

logging.getLogger().setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='TRPO.')
parser.add_argument("--task", type=str, default='Breakout-ram-v0')
parser.add_argument("--timesteps_per_batch", type=int, default=10000)
parser.add_argument("--n_steps", type=int, default=1000000000)
parser.add_argument("--n_iter", type=int, default=250)
parser.add_argument("--gamma", type=float, default=.995)
parser.add_argument("--max_kl", type=float, default=.1)
parser.add_argument("--cg_damping", type=float, default=1)
parser.add_argument("--lam", type=float, default=0.97)
parser.add_argument("--rollout_limit", type=str, default="episodes") # timesteps, episodes
parser.add_argument("--value_function_lr", type=float, default=.001)
parser.add_argument("--plot", type=bool, default=False)

args = parser.parse_args()
args.max_pathlength = gym.spec(args.task).timestep_limit

if (not args.timesteps_per_batch % comm.Get_size() == 0) and args.rollout_limit == "timesteps":
    print("*** Error: timesteps_per_batch must be divisible by number of processes when using equal timestep rollout")

env = gym.make(args.task)
mondir = os.getcwd() + ".dir"
if os.path.exists(mondir): shutil.rmtree(mondir)
os.mkdir(mondir)


n_extra_features = 1    # Add 1 for scaled timestep!
policy_model = DQNSoftmax(env.observation_space.shape[0], env.action_space.n)
value_function_model = DQNRegressor(env.observation_space.shape[0] + n_extra_features)
actor = TRPOActor(args, env, policy_model, value_function_model)
if rank == 0:
    learner = TRPOLearner(args, env, policy_model, value_function_model, actor.column_headings)
    new_policy_weights = learner.get_policy_weights()
    new_value_function_weights = learner.get_value_function_weights()

    print("Problem size: actions = {}, observations = {}, features = {}".format(env.action_space.n,env.observation_space.shape[0], env.observation_space.shape[0] + n_extra_features))
    print("Args: {}".format(args))
    sys.stdout.flush()
else:


    dummy_learner = TRPOLearner(args, env, policy_model, value_function_model, actor.column_headings)
    new_policy_weights = dummy_learner.get_policy_weights()
    new_value_function_weights = dummy_learner.get_value_function_weights()

history = {}
history["rollout_time"] = []
history["learn_time"] = []
history["bcast_time"] = []
history["gather_time"] = []
history["iteration_time"] = []
history["mean_reward"] = []
history["timesteps"] = []
history["episodes"] = []
history["delta_kl"] = []
history["surrogate_loss"] = []

totalsteps = 0
iteration = 0
is_done = 0
start_time = time.time()
decay = 'False'
while is_done == 0:
    seed_iter = itertools.count()
    iteration += 1

    # synchronize policy and vf model parameters and update actor weights locally
    bcast_start = time.time()
    comm.Bcast(new_policy_weights.data.numpy(), root=0)  # this modifies in place the numpy container for the weights
    comm.Bcast(new_value_function_weights.data.numpy(), root=0)
    actor.set_policy_weights(new_policy_weights)
    actor.set_value_function_weights(new_value_function_weights)
    bcast_time = (time.time() - bcast_start)



    # start worker processes collect experience for a minimum args.timesteps_per_batch timesteps
    rollout_start = time.time()
    data_paths, data_grads, data_rewards = actor.rollouts(args.timesteps_per_batch / comm.Get_size(), seed_iter)
    rollout_time = (time.time() - rollout_start)

    # gathering of experience on root process
    gather_start = time.time()
    paths, policy_gradients, episodes_rewards = gather_paths(data_paths, data_grads, data_rewards, args.rollout_limit)
    gather_time = (time.time() - gather_start)

    # only master process does learning


    if rank == 0:
        # learning step
        learn_start = time.time()
        if iteration == 150:
            decay = 'True'
            learner.args.max_kl = .01
            learner.args.cg_damping = .1

        stats, new_policy_weights, new_value_function_weights = learner.learn(paths, policy_gradients, episodes_rewards)
        learn_time = (time.time() - learn_start)
        iteration_time = rollout_time + learn_time + gather_time + bcast_time

        print(("\n-------- Iteration %d ----------" % iteration))
        print(("Reward Statistics:"))
        for k, v in stats.items():
            print("\t{} = {}".format(k,v))
        print(("Timing Statistics:"))
        print(("\tBroadcast time = %.3f s" % bcast_time))
        print(("\tRollout time = %.3f s" % rollout_time))
        print(("\tGather time = %.3f s" % gather_time))
        print(("\tLearn time = %.3f s" % learn_time))
        print(("\tTotal iteration time = %.3f s" % (rollout_time + learn_time + gather_time + bcast_time)))

        history["rollout_time"].append(rollout_time)
        history["learn_time"].append(learn_time)
        history["bcast_time"].append(bcast_time)
        history["gather_time"].append(gather_time)
        history["iteration_time"].append(rollout_time + learn_time + gather_time + bcast_time)
        history["mean_reward"].append(stats["Avg_Reward"])
        history["timesteps"].append(stats["Timesteps"])
        history["episodes"].append(stats['Episodes'])
        history["delta_kl"].append(stats["Delta_KL"])
        history["surrogate_loss"].append(stats["Surrogate loss"])

        # compute ~100 episode average reward
        ep = 0
        it = iteration-1
        rew = 0
        while ep < 100 and it >= 0:
            ep += history['episodes'][it]
            rew += history['mean_reward'][it]*history['episodes'][it]
            it -= 1
        if ep == 0:
            print("*** Problem: no complete episodes collected, increase timesteps_per_batch!")
            is_done = 1
        else:
            print(("Cumulative Reward Statistics:"))
            print(("\tMaximum Avg_reward = %.3f from iteration %d" % (np.max(history["mean_reward"]), 1+np.argmax(history["mean_reward"]))))
            print(("\tLast %d Episode Avg_reward = %.3f" % (ep, (rew / ep))))

        print(("Cumulative Mean Timing Statistics:"))
        print(("\tBroadcast time = %.3f s" % np.mean(history["bcast_time"])))
        print(("\tRollout time = %.3f s" % np.mean(history["rollout_time"])))
        print(("\tGather time = %.3f s" % np.mean(history["gather_time"])))
        print(("\tLearn time = %.3f s" % np.mean(history["learn_time"])))
        print(("\tTotal iteration time = %.3f s" % np.mean(history["iteration_time"])))

        if iteration % 10 == 0:
            with open("results/%s-%d-%f-%d" % (args.task, args.timesteps_per_batch, args.max_kl, comm.Get_size()), "w") as outfile:
                json.dump(history,outfile)

        if iteration > 2:
            if history['mean_reward'][-1] > np.max(history["mean_reward"][:-1]):
                print("Saving New Best model")
                checkpoint = "checkpoints/decay%s_task%s_timesteps%d_maxkl%f_comm%dvflr%d" % (decay, args.task, args.timesteps_per_batch, args.max_kl, comm.Get_size(), args.value_function_lr)
                save_checkpoint({
                    'epoch': iteration,
                    'policy_dict': learner.policy_net.state_dict(),
                    'mean_reward': history['mean_reward'][-1],
                    'episodes': history['episodes'][-1],
                    'delta_kl': history['delta_kl'][-1],
                    'surrogate_loss': history['surrogate_loss'][-1],
                    'history': history,
                    'args': args,
                    'value_dict': learner.value_net.state_dict()
                }, True, checkpoint=checkpoint)


            # learner.save_weights("{}-{}-{}-{}_{}.ckpt".format(args.task, args.timesteps_per_batch, args.max_kl, comm_size, iteration))  # XXX: not implemented yet!

        totalsteps += stats["Timesteps"]
        print(("%d total steps have happened (Elapsed time = %.3f s)" % (totalsteps, time.time() - start_time)))
        sys.stdout.flush()
        if iteration >= args.n_iter or totalsteps >= args.n_steps:
            is_done = 1


        if args.plot:
            if not iteration % 20:
                ob = env.reset()
                env.render()
                # for i in xrange(n_timesteps):
                for i in range(10000):
                    a, _ = actor.sample_action_from_policy(ob)
                    (ob, _rew, done, _info) = env.step(a.data[0,0].item())
                    env.render()
                    if done:
                        print("terminated after %s timesteps" % i)
                        break
                    time.sleep(.001)

    is_done = comm.bcast(is_done, root=0)


if rank == 0:
    print(("\n----- Evaluation complete! -----"))

