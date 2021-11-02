# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
from __future__ import print_function
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
        self.args.seq_len = 10
        self.args.horizon = 10
        self.args.grad_clip = 100.0
        self.args.rssm_type = 'continuous'
        self.args.kl_info = {'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0}
        self.args.loss_scale = {'kl':1, 'discount':10.0}
        

        self.args.rssm_info = {'deter_size':100, 'stoch_size':256, 'class_size':16, 'category_size':16, 'min_std':0.1} 
        self.args.rssm_node_size = 100
        self.args.rssm_embedding_size = 100
        # deter_size: rnn size
        self.RSSM = RSSM(self.args.max_num_limbs, self.args.rssm_node_size, 
                        self.args.rssm_embedding_size, device, 
                        self.args.rssm_type, self.args.rssm_info).to(device)


        self.args.obs_encoder = {'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU}
        input_size = self.args.limb_obs_size*self.args.max_num_limbs
        output_size = self.args.rssm_embedding_size
        self.ObsEncoder = DenseModel(input_size, output_size,  self.args.obs_encoder).to(device)

        self.args.obs_decoder = {'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU}
        modelstate_size = self.args.rssm_info['stoch_size'] + self.args.rssm_info['deter_size'] 
        input_size = modelstate_size
        output_size = self.args.limb_obs_size*self.args.max_num_limbs
        self.ObsDecoder = DenseModel(input_size, output_size, self.args.obs_decoder).to(device)

        self.args.var_disag = {'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU}
        input_size = self.args.rssm_info['deter_size']
        output_size = self.args.rssm_embedding_size
        self.var_networks = [DenseModel(input_size, output_size, self.args.var_disag).to(device) for _ in range(10)]
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.world_list = [self.RSSM, self.ObsEncoder, self.ObsDecoder]
        self.var_optimizer = torch.optim.Adam(get_parameters(self.var_networks), lr=args.lr)
        self.model_optimizer = torch.optim.Adam(get_parameters(self.world_list), lr=args.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

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

    def train_single(self, replay_buffer, iterations, batch_size, discount=0.99,
                tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):


            

            start_time = timer()
            # sample replay buffer
            x, y, u, r, d = replay_buffer.sample_seq_len(batch_size, self.args.seq_len)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)



            model_loss, posterior = self.representation_loss(next_state, action, done)
            
            end_time = timer()
            # print("[time] [place1]: ", end_time - start_time)
            
            
            t = RSSMContState(
                    posterior.mean[:,0],
                    posterior.std[:,0],
                    posterior.stoch[:,0], 
                    posterior.deter[:,0],
                )
            batched_posterior = self.RSSM.rssm_detach(t)
            
            start_time = timer()
            with FreezeParameters(self.world_list):
                _, disags_reward = self.RSSM.rollout_imagination(
                                        self.args.horizon, 
                                        self.actor, 
                                        batched_posterior,
                                        self.var_networks,
                                        self.ObsDecoder
                                 )
            end_time = timer()
            # print("[time] [place2]: ", end_time - start_time)
            disags_reward = torch.stack((disags_reward))
            disags_reward = disags_reward.swapaxes(0,1)
            disags_reward = disags_reward.mean(-1)
            disags_reward = disags_reward.unsqueeze(-1)
            disags_reward = torch.cuda.FloatTensor(disags_reward).to(device)

            self.model_optimizer.zero_grad()
            self.var_optimizer.zero_grad()
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                                get_parameters(self.world_list), 
                                self.args.grad_clip
                              )
            torch.nn.utils.clip_grad_norm_(
                                get_parameters(self.var_networks), 
                                self.args.grad_clip
                              )
            self.model_optimizer.step()
            self.var_optimizer.step()

            start_time = timer()
            x, y, u, r, d = x[:,0], y[:,0], u[:,0], r[:,0], d[:,0]
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            # reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            # select action according to policy and add clipped noise
            with torch.no_grad():
                noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = self.actor_target(next_state) + noise
                next_action = next_action.clamp(-self.args.max_action, self.args.max_action)

                # Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = disags_reward + (done * discount * target_Q)

            # get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # delayed policy updates
            if it % policy_freq == 0:

                # compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

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
            end_time = timer()
            # print("[time] [place3]: ", end_time - start_time)

    def train(self, replay_buffer_list, iterations_list, batch_size, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, graphs=None, envs_train_names=None):
        per_morph_iter = sum(iterations_list) // len(envs_train_names)
        for env_name in envs_train_names:
            replay_buffer = replay_buffer_list[env_name]
            self.change_morphology(graphs[env_name])
            self.train_single(replay_buffer, per_morph_iter, batch_size=batch_size, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

    def save(self, fname):
        torch.save(self.RSSM.state_dict(), '%s_RSSM.pth' % fname)
        torch.save(self.actor.state_dict(), '%s_actor.pth' % fname)
        torch.save(self.critic.state_dict(), '%s_critic.pth' % fname)

    def load(self, fname):
        self.RSSM.load_state_dict(torch.load('%s_RSSM.pth' % fname))
        self.actor.load_state_dict(torch.load('%s_actor.pth' % fname))
        self.critic.load_state_dict(torch.load('%s_critic.pth' % fname))

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
        prior, posterior, preds_disag = self.RSSM.rollout_observation(self.args.seq_len, embed, action_x, done, prev_rssm_state, self.var_networks)
        post_modelstate = self.RSSM.get_model_state(posterior)   
        obs_decode = self.ObsDecoder(post_modelstate)
        obs_loss = self._obs_loss(obs_decode, obs)
        preds_loss = self._pred_loss(preds_disag, embed)
        _, _, kl_loss = self._kl_loss(prior, posterior)

        model_loss = kl_loss + obs_loss + preds_loss
        return model_loss, posterior
    
    def _pred_loss(self, pred_disag, embed):
        pred_disag = pred_disag.swapaxes(0,2)
        embed = embed[:,1:,:]
        embed = embed.unsqueeze(1).repeat(1, pred_disag.shape[1], 1, 1)
        pred_loss = nn.L1Loss()(pred_disag,embed)
        return pred_loss

    def _obs_loss(self, obs_decode, obs):
        obs = obs[:,1:,:]
        obs_decode = obs_decode[:,:,:obs.shape[2]]
        obs_loss = nn.L1Loss()(obs_decode, obs)
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
    