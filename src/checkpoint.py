from __future__ import print_function
import os
import torch
import utils
import numpy as np


def has_checkpoint(checkpoint_path, rb_path):
    """check if a checkpoint exists"""
    if not (os.path.exists(checkpoint_path) and os.path.exists(rb_path)):
        return False
    if 'model.pyth' not in os.listdir(checkpoint_path):
        return False
    if len(os.listdir(rb_path)) == 0:
        return False
    return True


def save_model(checkpoint_path, real_policy, expl_policy, wm, total_timesteps, total_train_timestep_list, episode_num, num_samples, replay_buffer, env_names, args):
    # change to default graph before saving
    real_policy.change_morphology([-1])
    expl_policy.change_morphology([-1])
    # Record the state
    checkpoint = {
        'actor_state': real_policy.actor.state_dict(),
        'critic_state': real_policy.critic.state_dict(),
        'actor_target_state': real_policy.actor_target.state_dict(),
        'critic_target_state': real_policy.critic_target.state_dict(),
        'actor_optimizer_state': real_policy.actor_optimizer.state_dict(),
        'critic_optimizer_state': real_policy.critic_optimizer.state_dict(),
        
        'expl_actor_state': expl_policy.actor.state_dict(),
        'expl_critic_state': expl_policy.critic.state_dict(),
        'expl_actor_target_state': expl_policy.actor_target.state_dict(),
        'expl_critic_target_state': expl_policy.critic_target.state_dict(),
        'expl_actor_optimizer_state': expl_policy.actor_optimizer.state_dict(),
        'expl_critic_optimizer_state': expl_policy.critic_optimizer.state_dict(),
        
        'wm_RSSM': wm.RSSM.state_dict(),
        'wm_ObsEncoder': wm.ObsEncoder.state_dict(),
        'wm_ObsDecoder': wm.ObsDecoder.state_dict(),
        'wm_RewardDecoder': wm.RewardDecoder.state_dict(),
        'wm_DiscountModel' : wm.DiscountModel.state_dict(),
        "wm_var_networks" : {i : wm.var_networks[i].state_dict() for i in range(len(wm.var_networks)) },
        'wm_model_optimizer': wm.model_optimizer.state_dict(),
        'wm_var_optimizer': wm.var_optimizer.state_dict(),

        'total_timesteps': total_timesteps,
        'total_train_timestep_list': total_train_timestep_list,
        'episode_num': episode_num,
        'num_samples': num_samples,
        'args': real_policy.args,
        'expl_args': expl_policy.args,
        'wm_args': wm.args,
        'rb_max': {name: replay_buffer[name].max_size for name in replay_buffer},
        'rb_ptr': {name: replay_buffer[name].ptr for name in replay_buffer},
        'rb_slicing_size': {name: replay_buffer[name].slicing_size for name in replay_buffer}
    }
    fpath = os.path.join(checkpoint_path, 'model.pyth')
    # (over)write the checkpoint
    torch.save(checkpoint, fpath)
    return fpath


def save_replay_buffer(rb_path, replay_buffer):
    # save replay buffer
    for name in replay_buffer:
        np.save(os.path.join(rb_path, '{}.npy'.format(name)), np.array(replay_buffer[name].storage), allow_pickle=False)
    return rb_path


def load_checkpoint(load_exp_path, rb_path, real_policy, expl_policy, wm, args):
    fpath = os.path.join(load_exp_path, 'model.pyth')
    checkpoint = torch.load(fpath, map_location='cpu')
    # change to default graph before loading
    real_policy.change_morphology([-1])
    # load and return checkpoint
    real_policy.actor.load_state_dict(checkpoint['actor_state'])
    real_policy.critic.load_state_dict(checkpoint['critic_state'])
    real_policy.actor_target.load_state_dict(checkpoint['actor_target_state'])
    real_policy.critic_target.load_state_dict(checkpoint['critic_target_state'])
    real_policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
    real_policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
    real_policy.args = checkpoint['args']
    
    if 'expl_actor_state' in checkpoint.keys():
        print("load expl and world model")
        expl_policy.change_morphology([-1])
        expl_policy.actor.load_state_dict(checkpoint['expl_actor_state'])
        expl_policy.critic.load_state_dict(checkpoint['expl_critic_state'])
        expl_policy.actor_target.load_state_dict(checkpoint['expl_actor_target_state'])
        expl_policy.critic_target.load_state_dict(checkpoint['expl_critic_target_state'])
        expl_policy.actor_optimizer.load_state_dict(checkpoint['expl_actor_optimizer_state'])
        expl_policy.critic_optimizer.load_state_dict(checkpoint['expl_critic_optimizer_state'])
        expl_policy.args = checkpoint['expl_args']

        wm.RSSM.load_state_dict(checkpoint['wm_RSSM'])
        wm.ObsEncoder.load_state_dict(checkpoint['wm_ObsEncoder'])
        wm.ObsDecoder.load_state_dict(checkpoint['wm_ObsDecoder'])
        wm.RewardDecoder.load_state_dict(checkpoint['wm_RewardDecoder'])
        wm.model_optimizer.load_state_dict(checkpoint['wm_model_optimizer'])
        wm.var_optimizer.load_state_dict(checkpoint['wm_var_optimizer'])
        wm.DiscountModel.load_state_dict(checkpoint['wm_DiscountModel'])

        for i in range(10):
            wm.var_networks[i].load_state_dict(checkpoint['wm_var_networks'][i])
            
        wm.args = checkpoint['wm_args']

    # load replay buffer
    all_rb_files = [f[:-4] for f in os.listdir(rb_path) if '.npy' in f]
    all_rb_files.sort()
    replay_buffer_new = dict()
    for name in all_rb_files:
        if len(all_rb_files) > args.rb_max // 1e6:
            replay_buffer_new[name] = utils.ReplayBuffer(max_size=args.rb_max // len(all_rb_files))
        else:
            replay_buffer_new[name] = utils.ReplayBuffer()
        replay_buffer_new[name].max_size = int(checkpoint['rb_max'][name])
        replay_buffer_new[name].ptr = int(checkpoint['rb_ptr'][name])
        replay_buffer_new[name].slicing_size = checkpoint['rb_slicing_size'][name]
        replay_buffer_new[name].storage = list(np.load(os.path.join(rb_path, '{}.npy'.format(name))))

    return checkpoint['total_timesteps'], \
            checkpoint['total_train_timestep_list'], \
            checkpoint['episode_num'], \
            replay_buffer_new, \
            checkpoint['num_samples'], \
            fpath


def load_model_only(exp_path, real_policy ,expl_policy, vis=False):
    model_path = os.path.join(exp_path, 'model.pyth')
    if not os.path.exists(model_path):
        raise FileNotFoundError('no model file found')
    print('*** using model {} ***'.format(model_path))
    checkpoint = torch.load(model_path, map_location='cpu')
    # change to default graph before loading
    real_policy.change_morphology([-1])
    # load and return checkpoint
    real_policy.actor.load_state_dict(checkpoint['actor_state'])
    real_policy.critic.load_state_dict(checkpoint['critic_state'])
    real_policy.actor_target.load_state_dict(checkpoint['actor_target_state'])
    real_policy.critic_target.load_state_dict(checkpoint['critic_target_state'])
    real_policy.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
    real_policy.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
    real_policy.args = checkpoint['args']


    if 'expl_actor_state' in checkpoint.keys():
        print("load expl model")
        expl_policy.change_morphology([-1])
        expl_policy.actor.load_state_dict(checkpoint['expl_actor_state'])
        expl_policy.critic.load_state_dict(checkpoint['expl_critic_state'])
        expl_policy.actor_target.load_state_dict(checkpoint['expl_actor_target_state'])
        expl_policy.critic_target.load_state_dict(checkpoint['expl_critic_target_state'])
        expl_policy.actor_optimizer.load_state_dict(checkpoint['expl_actor_optimizer_state'])
        expl_policy.critic_optimizer.load_state_dict(checkpoint['expl_critic_optimizer_state'])
        expl_policy.args = checkpoint['expl_args']