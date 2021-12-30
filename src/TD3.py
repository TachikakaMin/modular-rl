# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
from __future__ import print_function
from enum import EnumMeta
import torch
import torch.nn.functional as F
from ModularActor import ActorGraphPolicy,DisagreeMLP
from ModularCritic import CriticGraphPolicy
from rssm import RSSM
from rssm_util import get_parameters,FreezeParameters, RSSMContState
from dense import DenseModel
import numpy as np
import time
import torch.nn as nn
from timeit import default_timer as timer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):

    def __init__(self, args):

        self.args = args
        self.isExpl = args.isExpl
        self.actor = ActorGraphPolicy(self.args.limb_obs_size, 1,
                                      self.args.msg_dim, self.args.batch_size,
                                      self.args.max_action, self.args.max_children,
                                      self.args.disable_fold, self.args.td, self.args.bu).to(device)
        self.actor_target = ActorGraphPolicy(self.args.limb_obs_size, 1,
                                             self.args.msg_dim, self.args.batch_size,
                                             self.args.max_action, self.args.max_children,
                                             self.args.disable_fold, self.args.td, self.args.bu).to(device)
        self.critic = CriticGraphPolicy(self.args.limb_obs_size, 1,
                                        self.args.msg_dim, self.args.batch_size,
                                        self.args.max_children, self.args.disable_fold,
                                        self.args.td, self.args.bu).to(device)
        self.critic_target = CriticGraphPolicy(self.args.limb_obs_size, 1,
                                               self.args.msg_dim, self.args.batch_size,
                                               self.args.max_children, self.args.disable_fold,
                                               self.args.td, self.args.bu).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        if args.isExpl:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        else :
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr*0.1)

    def change_morphology(self, graph):
        self.actor.change_morphology(graph)
        self.actor_target.change_morphology(graph)
        self.critic.change_morphology(graph)
        self.critic_target.change_morphology(graph)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state, 'inference').cpu().numpy().flatten()
            return action

    def train_single(self, wm, log_var, replay_buffer, iterations, batch_size, discount=0.99,
                tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            env_index = log_var["index"]
            timestep = log_var["total_train_timestep_list"][env_index]
            log_var["total_train_timestep_list"][env_index] = timestep + 1
            writer = log_var["writer"]
            env_name = log_var["env_name"]
            # sample replay buffer
            x, y, u, r, d  = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            real_reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            if self.isExpl:
                next_rssm_states, disag_reward, action_rssm, tot_reward = wm.rollout_imagination(self.actor, state, useDis = True)
                tot_reward = tot_reward*done
                writer.add_scalar('{}_expl_reward'.format(env_name), tot_reward.mean(0).item(), timestep)    
            else :
                next_rssm_states, disag_reward, action_rssm, tot_reward = wm.rollout_imagination(self.actor, state, useDis = False)
                tot_reward = tot_reward*done
                writer.add_scalar('{}_pseudo_reward'.format(env_name), tot_reward.mean(0).item(), timestep)    
            
            
            next_state = state 
            for h in range(wm.horizon):
                with torch.no_grad():
                    last_state = next_state
                    reward = disag_reward[h].detach()
                    next_state = next_rssm_states[h].detach()
                    action = action_rssm[h].detach()

                # select action according to policy and add clipped noise
                with torch.no_grad():
                    noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                    noise = noise.clamp(-noise_clip, noise_clip)
                    next_action = self.actor_target(next_state) + noise
                    next_action = next_action.clamp(-self.args.max_action, self.args.max_action)

                    # Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + (done * discount * target_Q)

                # get current Q estimates
                current_Q1, current_Q2 = self.critic(last_state, action)

                # compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                if self.isExpl:
                    writer.add_scalar('{}_expl_critic_loss'.format(env_name), critic_loss.item(), timestep*wm.horizon + h)
                else:
                    writer.add_scalar('{}_critic_loss'.format(env_name), critic_loss.item(), timestep*wm.horizon + h)
                # optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # delayed policy updates
                if (it*wm.horizon + h) % policy_freq == 0:

                    # compute actor loss
                    actor_loss = -self.critic.Q1(last_state, self.actor(last_state)).mean()
                    if self.isExpl:
                        writer.add_scalar('{}_expl_actor_loss'.format(env_name), actor_loss.item(), timestep*wm.horizon + h)
                    else :
                        writer.add_scalar('{}_actor_loss'.format(env_name), actor_loss.item(), timestep*wm.horizon + h)
                
                    # optimize the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the frozen target models
                    for param, target_param in zip(self.critic.parameters(),
                                                self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(),
                                                self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def train(self, wm, log_var, replay_buffer_list, iterations_list, batch_size, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, graphs=None, envs_train_names=None):
        per_morph_iter = sum(iterations_list) // len(envs_train_names)
        for i, env_name in enumerate(envs_train_names):
            log_var["env_name"] = env_name
            log_var["index"] = i
            replay_buffer = replay_buffer_list[env_name]
            self.change_morphology(graphs[env_name])
            self.train_single(wm, log_var, replay_buffer, per_morph_iter, batch_size=batch_size, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

    def save(self, fname):
        torch.save(self.RSSM.state_dict(), '%s_RSSM.pth' % fname)
        torch.save(self.actor.state_dict(), '%s_actor.pth' % fname)
        torch.save(self.critic.state_dict(), '%s_critic.pth' % fname)
        torch.save(self.ObsEncoder.state_dict(), '%s_ObsEncoder.pth' % fname)
        torch.save(self.ObsDecoder.state_dict(), '%s_ObsDecoder.pth' % fname)
        for i,var_network in enumerate(self.var_networks):
            torch.save(var_network.state_dict(), '%s_var_network_%s.pth' % (fname, str(i)))


    def load(self, fname):
        self.RSSM.load_state_dict(torch.load('%s_RSSM.pth' % fname))
        self.actor.load_state_dict(torch.load('%s_actor.pth' % fname))
        self.critic.load_state_dict(torch.load('%s_critic.pth' % fname))
        self.ObsEncoder.load_state_dict(torch.load('%s_ObsEncoder.pth' % fname))
        self.ObsDecoder.load_state_dict(torch.load('%s_ObsDecoder.pth' % fname))
        for i,_ in enumerate(self.var_networks):
            self.var_networks[i].load_state_dict(torch.load('%s_var_network_%s.pth' % (fname, str(i))))
        
    def representation_loss(self, obs, action, done):

        input_size = self.args.limb_obs_size*self.args.max_num_limbs
        obs_x = torch.zeros(obs.shape[0], obs.shape[1], input_size)
        obs_x[:,:,:obs.shape[2]] = obs
        obs_x = obs_x.to(device)


        embed = self.ObsEncoder(obs_x)

        action_x = torch.zeros(action.shape[0], action.shape[1], self.args.max_num_limbs)
        action_x[:,:,:action.shape[2]] = action
        action_x = action_x.to(device)


        prev_rssm_state = self.RSSM._init_rssm_state(self.args.batch_size)   
        prior, posterior, preds_disag_dist = self.RSSM.rollout_observation(self.args.seq_len, embed, action_x, done, prev_rssm_state, self.var_networks)
        post_modelstate = self.RSSM.get_model_state(posterior)   
        obs_dist = self.ObsDecoder(post_modelstate)
        obs_loss = self._obs_loss(obs_dist, obs)

        preds_loss = self._pred_loss(preds_disag_dist, embed)
        prior_dist, post_dist, kl_loss = self._kl_loss(prior, posterior)

        model_loss = kl_loss + obs_loss + preds_loss
        return model_loss, kl_loss, obs_loss, preds_loss, posterior, prior_dist, post_dist
    
    def _pred_loss(self, pred_disag_dist, embed):
        pred_loss = -torch.mean(pred_disag_dist.log_prob(embed))
        return pred_loss

    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.args.kl_info['use_kl_balance']:
            alpha = self.args.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.args.kl_info['use_free_nats']:
                free_nats = self.args.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.args.kl_info['use_free_nats']:
                free_nats = self.args.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    