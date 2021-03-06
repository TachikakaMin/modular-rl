from __future__ import print_function
import numpy as np
import torch
import os
import utils
import cv2
import TD3
# import TD3_bak as TD3
from copy import deepcopy as dp
import json
import time
from tensorboardX import SummaryWriter
from arguments import get_args
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import checkpoint as cp
from config import *
from dataclasses import dataclass, field
import torch.nn as nn
from new_model.wm import WorldModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def make_paths(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)
    exp_name = "EXP_%04d" % (args.expID)
    exp_path = os.path.join(DATA_DIR, exp_name)
    rb_path = os.path.join(BUFFER_DIR, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(rb_path, exist_ok=True)
    # save arguments
    with open(os.path.join(exp_path, 'args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    load_exp_name = "EXP_%04d" % (args.loadID)
    load_exp_path = os.path.join(DATA_DIR, load_exp_name)
    load_rb_path = os.path.join(BUFFER_DIR, load_exp_name)

    return args, exp_path, rb_path, load_exp_path, load_rb_path, exp_name


def make_envs(args):
    # Retrieve MuJoCo XML files for training ========================================
    envs_train_names = []
    args.graphs = dict()
    # existing envs
    if not args.custom_xml:
        for morphology in args.morphologies:
            envs_train_names += [name[:-4] for name in os.listdir(XML_DIR) if '.xml' in name and morphology in name]

        for name in envs_train_names:
            args.graphs[name] = utils.getGraphStructure(os.path.join(XML_DIR, '{}.xml'.format(name)))
    # custom envs
    else:
        if os.path.isfile(args.custom_xml):
            assert '.xml' in os.path.basename(args.custom_xml), 'No XML file found.'
            name = os.path.basename(args.custom_xml)
            envs_train_names.append(name[:-4])  # truncate the .xml suffix
            args.graphs[name[:-4]] = utils.getGraphStructure(args.custom_xml)
        elif os.path.isdir(args.custom_xml):
            for name in os.listdir(args.custom_xml):
                if '.xml' in name:
                    envs_train_names.append(name[:-4])
                    args.graphs[name[:-4]] = utils.getGraphStructure(os.path.join(args.custom_xml, name))
    envs_train_names.sort()
    num_envs_train = len(envs_train_names)
    args.num_envs_train = num_envs_train
    print("#" * 50 + '\ntraining envs: {}\n'.format(envs_train_names) + "#" * 50)

    # Set up training env and policy ================================================
    args.limb_obs_size, args.max_action, envs_train = utils.registerEnvs(envs_train_names, args)
    max_num_limbs = max([len(args.graphs[env_name]) for env_name in envs_train_names])
    args.max_num_limbs = max_num_limbs
    
    
    # create vectorized training env
    obs_max_len = max([len(args.graphs[env_name]) for env_name in envs_train_names]) * args.limb_obs_size
    args.obs_max_len = obs_max_len
    envs_train = SubprocVecEnv(envs_train)  # vectorized env
    # determine the maximum number of children in all the training envs
    if args.max_children is None:
        args.max_children = utils.findMaxChildren(envs_train_names, args.graphs)
    
    return args, envs_train, num_envs_train, envs_train_names, max_num_limbs


def make_models(args):
    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # setup agent policy
    args.isExpl = True
    expl_policy = TD3.TD3(args)
    args.isExpl = False
    real_policy = TD3.TD3(args)

    args.img_shape = (3, 64, 64)
    args.var_disag = {'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU}
    args.seq_len = 2
    args.horizon = 2
    args.grad_clip = 100.0
    wm = WorldModel(encode_size = 128, args = args)
    return args, expl_policy, real_policy, wm


def make_replay_buffer(args, envs_train_names):
    replay_buffer = dict()
    if args.num_envs_train > args.rb_max // 1e6:
        for name in envs_train_names:
            replay_buffer[name] = utils.ReplayBuffer(max_size=args.rb_max // num_envs_train)
    else:
        for name in envs_train_names:
            replay_buffer[name] = utils.ReplayBuffer()
    return args, replay_buffer


def load_models(args, replay_buffer, load_exp_path, load_rb_path, real_policy, expl_policy, wm):
    total_train_timestep_list = [0 for i in range(args.num_envs_train)]
    total_timesteps = 0
    episode_num = 0
    num_samples = 0
    if cp.has_checkpoint(load_exp_path, load_rb_path):
        print("*** loading checkpoint from {} ***".format(load_exp_path))
        if args.only_load_model:
            print("only load model")
            _, _, _, _, _, loaded_path = cp.load_checkpoint(load_exp_path, load_rb_path, real_policy, expl_policy, wm, args)
        else:
            total_timesteps, total_train_timestep_list, episode_num, replay_buffer, num_samples, loaded_path = cp.load_checkpoint(load_exp_path, load_rb_path, real_policy, expl_policy, wm, args)
        print("*** checkpoint loaded from {} ***".format(loaded_path))
    return args, total_timesteps, total_train_timestep_list, episode_num, replay_buffer, num_samples

def train(args):

    # Set up directories ===========================================================
    args, exp_path, rb_path, load_exp_path, load_rb_path, exp_name = make_paths(args)

    args, envs_train, num_envs_train, envs_train_names, max_num_limbs = make_envs(args)
    
    args, replay_buffer = make_replay_buffer(args, envs_train_names)

    args, expl_policy, real_policy, wm = make_models(args)

        
    print("*** training from scratch ***")
    # Initialize training variables ================================================
    writer = SummaryWriter("%s/%s/" % (DATA_DIR, exp_name))
    s = time.time()
    timesteps_since_saving = 0
    timesteps_since_test = 0
    timesteps_since_saving_model_only = 0
    this_training_timesteps = 0
    collect_done = False
    obs_list = envs_train.reset()
    done_list = [False for i in range(num_envs_train)]
    episode_reward_list = [0 for i in range(num_envs_train)]
    episode_timesteps_list = [0 for i in range(num_envs_train)]
    episode_reward_list_buffer = [0 for i in range(num_envs_train)]
    
    (args, total_timesteps, total_train_timestep_list, 
                episode_num, replay_buffer, num_samples) = load_models(args, replay_buffer, load_exp_path, load_rb_path, real_policy, expl_policy, wm)
    

    # Start training ===========================================================
    while total_timesteps < args.max_timesteps:
        # train and log after one episode for each env
        if collect_done:
            # log updates and train policy
            print("train world model")
            log_var = {"writer": writer, "total_train_timestep_list": dp(total_train_timestep_list)}
            wm.train(log_var, episode_timesteps_list, args.batch_size, replay_buffer, envs_train_names=envs_train_names[:num_envs_train])

            print("train expl_policy")
            # log_var = {"writer": writer, "total_train_timestep_list": dp(total_train_timestep_list)}
            log_var = {"writer": writer, "total_train_timestep_list": total_train_timestep_list}
            expl_policy.train(wm, log_var, replay_buffer, episode_timesteps_list, args.batch_size,
                        args.discount, args.tau, args.policy_noise, args.noise_clip,
                        args.policy_freq, graphs=args.graphs, envs_train_names=envs_train_names[:num_envs_train])
            
            # print("train real_policy")
            # log_var = {"writer": writer, "total_train_timestep_list": total_train_timestep_list}
            # real_policy.train(wm, log_var, replay_buffer, episode_timesteps_list, args.batch_size,
            #             args.discount, args.tau, args.policy_noise, args.noise_clip,
            #             args.policy_freq, graphs=args.graphs, envs_train_names=envs_train_names[:num_envs_train])
            

            # add to tensorboard display
            for i in range(num_envs_train):
                writer.add_scalar('{}_expl_episode_reward'.format(envs_train_names[i]), episode_reward_list[i], total_timesteps)
                writer.add_scalar('{}_expl_episode_len'.format(envs_train_names[i]), episode_timesteps_list[i], total_timesteps)
            # print to console
            print("-" * 50 + "\nExpID: {}, FPS: {:.2f}, TotalT: {}, EpisodeNum: {}, SampleNum: {}, ReplayBSize: {}".format(
                    args.expID, this_training_timesteps / (time.time() - s),
                    total_timesteps, episode_num, num_samples,
                    sum([len(replay_buffer[name].storage) for name in envs_train_names])))
            for i in range(len(envs_train_names)):
                print("{} === EpisodeT: {}, Reward: {:.2f}".format(envs_train_names[i],
                                                                    episode_timesteps_list[i],
                                                                    episode_reward_list[i]))

            # save model and replay buffers
            if timesteps_since_saving >= args.save_freq:
                timesteps_since_saving = 0
                model_saved_path = cp.save_model(exp_path, real_policy, expl_policy, wm,
                                                 total_timesteps, total_train_timestep_list,
                                                 episode_num, num_samples, replay_buffer,
                                                 envs_train_names, args)
                print("*** model saved to {} ***".format(model_saved_path))
                rb_saved_path = cp.save_replay_buffer(rb_path, replay_buffer)
                print("*** replay buffers saved to {} ***".format(rb_saved_path))

            # reset training variables
            obs_list = envs_train.reset()
            done_list = [False for i in range(num_envs_train)]
            episode_reward_list = [0 for i in range(num_envs_train)]
            episode_timesteps_list = [0 for i in range(num_envs_train)]
            episode_num += num_envs_train
            # create reward buffer to store reward for one sub-env when it is not done
            episode_reward_list_buffer = [0 for i in range(num_envs_train)]
      

        # start sampling ===========================================================
        # sample action randomly for sometime and then according to the policy

        if total_timesteps < args.start_timesteps:
            action_list = [np.random.uniform(low=envs_train.action_space.low[0],
                                             high=envs_train.action_space.high[0],
                                             size=max_num_limbs) for i in range(num_envs_train)]
        else:
            action_list = []
            for i in range(num_envs_train):
                # dynamically change the graph structure of the modular policy
                expl_policy.change_morphology(args.graphs[envs_train_names[i]])
                # remove 0 padding of obs before feeding into the policy (trick for vectorized env)
                obs = np.array(obs_list[i][:args.limb_obs_size * len(args.graphs[envs_train_names[i]])])
                policy_action = expl_policy.select_action(obs)
                if args.expl_noise != 0:
                    policy_action = (policy_action + np.random.normal(0, args.expl_noise,
                        size=policy_action.size)).clip(envs_train.action_space.low[0],
                        envs_train.action_space.high[0])
                # add 0-padding to ensure that size is the same for all envs
                policy_action = np.append(policy_action, np.array([0 for i in range(max_num_limbs - policy_action.size)]))
                action_list.append(policy_action)

        # perform action in the environment
        new_obs_list, reward_list, curr_done_list, _ = envs_train.step(action_list)
        env_img_list = envs_train.get_images()
        new_img_list = [cv2.resize(img, (64,64)) for img in env_img_list]
        # record if each env has ever been 'done'
        done_list = [done_list[i] or curr_done_list[i] for i in range(num_envs_train)]

        for i in range(num_envs_train):
            # add the instant reward to the cumulative buffer
            # if any sub-env is done at the momoent, set the episode reward list to be the value in the buffer
            episode_reward_list_buffer[i] += reward_list[i]
            if curr_done_list[i] and episode_reward_list[i] == 0:
                episode_reward_list[i] = episode_reward_list_buffer[i]
                episode_reward_list_buffer[i] = 0
            writer.add_scalar('{}_expl_instant_reward'.format(envs_train_names[i]), reward_list[i], total_timesteps)
            done_bool = float(curr_done_list[i])
            if episode_timesteps_list[i] + 1 == args.max_episode_steps:
                done_bool = 0
                done_list[i] = True
            # remove 0 padding before storing in the replay buffer (trick for vectorized env)
            num_limbs = len(args.graphs[envs_train_names[i]])
            obs = np.array(obs_list[i][:args.limb_obs_size * num_limbs])
            new_obs = np.array(new_obs_list[i][:args.limb_obs_size * num_limbs])
            action = np.array(action_list[i][:num_limbs])
            new_obs_img = new_img_list[i]
            # new_obs_img = 0
            
           
            replay_buffer[envs_train_names[i]].add((obs, new_obs, action, reward_list[i], done_bool, new_obs_img))
            num_samples += 1
            # do not increment episode_timesteps if the sub-env has been 'done'
            if not done_list[i]:
                episode_timesteps_list[i] += 1
                total_timesteps += 1
                this_training_timesteps += 1
                timesteps_since_saving += 1
                timesteps_since_test += 1
                timesteps_since_saving_model_only += 1

        obs_list = new_obs_list
        collect_done = all(done_list)

    # save checkpoint after training ===========================================================
    model_saved_path = cp.save_model(exp_path, expl_policy, total_timesteps, total_train_timestep_list, 
                                     episode_num, num_samples, replay_buffer,
                                     envs_train_names, args)
    print("*** training finished and model saved to {} ***".format(model_saved_path))


if __name__ == "__main__":
    args = get_args()
    print("device: ", device)
    train(args)
