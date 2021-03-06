# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
from __future__ import print_function
import torch
import torch.nn.functional as F
from ModularActor import ActorGraphPolicy
from ModularCritic import CriticGraphPolicy
import numpy as np
import time
import torch.nn as nn
from timeit import default_timer as timer
from new_model.module import get_parameters, FreezeParameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class TD3(object):

    def __init__(self, args):

        self.args = args
        self.isExpl = args.isExpl
        self.action_size = self.args.max_num_limbs
        self.state_size = self.args.limb_obs_size*self.args.max_num_limbs
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

    def train_single(self, wm, log_var, replay_buffer, iterations, batch_size, discount=0.99,
                tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        # print("[iterations]: ", iterations)
        for it in range(iterations):
            # print(it)
            env_index = log_var["index"]
            timestep = log_var["total_train_timestep_list"][env_index]
            log_var["total_train_timestep_list"][env_index] = timestep + 1
            writer = log_var["writer"]
            env_name = log_var["env_name"]
            # sample replay buffer

            x, y, u, r, d, imgs = replay_buffer.sample_seq_len(batch_size, self.args.horizon + 1)
            states = torch.FloatTensor(x).to(device)
            states = torch.permute(states, (1, 0, 2))
            modular_state_size = states.shape[2]
            obs = states[1:]
            obs_tmp = torch.zeros(self.args.horizon , batch_size, self.state_size).to(device)
            obs_tmp[:, :, :modular_state_size] = obs
            obs = obs_tmp

            actions = torch.FloatTensor(u).to(device)
            actions = torch.permute(actions, (1, 0, 2))
            modular_action_size = actions.shape[2]
            acts = actions[:-1]
            acts_tmp = torch.zeros(self.args.horizon, batch_size, self.action_size).to(device)
            acts_tmp[:, :, :modular_action_size] = acts
            acts = acts_tmp

            real_rewards = torch.FloatTensor(r).to(device)
            real_rewards = torch.permute(real_rewards, (1, 0, 2))
            rewards = real_rewards[:-1]

            dones = torch.FloatTensor(1 - d).to(device)
            dones = torch.permute(dones, (1, 0, 2)) # (horizon, bz, size)
            nonterms = dones[:-1]
            with torch.no_grad():
                embed = wm.ObsEncoder(obs)                                         #t to t+seq_len   
                prev_rssm_state = wm.RSSM._init_rssm_state(batch_size)   
                prior, posterior, action_state_embeds = wm.RSSM.rollout_observation(wm.seq_len, embed, acts, nonterms, prev_rssm_state)

            with torch.no_grad():
                batched_posterior = wm.RSSM.rssm_detach(wm.RSSM.rssm_seq_to_batch(posterior, batch_size, wm.seq_len-1))
            
            with FreezeParameters(wm.world_list):
                imag_rssm_states, imag_actions, state_action_embeds = wm.RSSM.rollout_imagination(wm.horizon, self.actor, modular_state_size, batched_posterior, wm.ObsDecoder)
                
            imag_modelstates = wm.RSSM.get_model_state(imag_rssm_states)
            with FreezeParameters(wm.world_list):
                # disag_reward = wm.get_disag_reward(state_action_embeds)
                # disag_reward = torch.unsqueeze(disag_reward, dim=-1)
                imag_reward_dist = wm.RewardDecoder(imag_modelstates)
                imag_reward = imag_reward_dist.mean
                discount_dist = wm.DiscountModel(imag_modelstates)
                discount_arr = wm.discount*torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)
                


            for h in range(wm.horizon-1):
                with torch.no_grad():
                    last_state = imag_modelstates[h].detach()
                    next_state = imag_modelstates[h+1].detach()
                    with FreezeParameters([wm.ObsDecoder]):
                        last_state = wm.ObsDecoder(last_state)   
                        last_state = last_state.mean[:, :modular_state_size]
                        next_state = wm.ObsDecoder(next_state)   
                        next_state = next_state.mean[:, :modular_state_size]

                    # if self.isExpl:
                    #     reward = disag_reward[h+1].detach()
                    # else:
                    #     reward = imag_reward[h+1].detach()
                    reward = imag_reward[h+1].detach()
                    
                    action = imag_actions[h+1].detach()[:, :modular_action_size]
                    done = discount_arr[h+1].detach()
                    done = torch.clamp(done, 0, 1)
                
                new_bz = next_state.shape[0]
                
                # select action according to policy and add clipped noise
                with torch.no_grad():
                    u = action.cpu()
                    noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                    noise = noise.clamp(-noise_clip, noise_clip)
                    next_action = self.actor_target(next_state, tmp_bs = new_bz) + noise
                    next_action = next_action.clamp(-self.args.max_action, self.args.max_action)

                    # Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action, tmp_bs = new_bz)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + (done * discount * target_Q)

                # get current Q estimates
                current_Q1, current_Q2 = self.critic(last_state, action, tmp_bs = new_bz)

                # compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                if self.isExpl:
                    writer.add_scalar('{}_expl_critic_loss'.format(env_name), critic_loss.item(), timestep*wm.horizon + h)
                    writer.add_scalar('{}_expl_not_done'.format(env_name), done[0].item(), timestep*wm.horizon + h)
                    writer.add_scalar('{}_wm_dis_reward'.format(env_name), reward.mean(), timestep*wm.horizon + h)
                else:
                    writer.add_scalar('{}_critic_loss'.format(env_name), critic_loss.item(), timestep*wm.horizon + h)
                    writer.add_scalar('{}_not_done'.format(env_name), done[0].item(), timestep*wm.horizon + h)
                    writer.add_scalar('{}_wm_reward'.format(env_name), reward.mean(), timestep*wm.horizon + h)
                # optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # delayed policy updates
                if (it*wm.horizon + h) % policy_freq == 0:

                    # compute actor loss
                    actor_loss = -self.critic.Q1(last_state, self.actor(last_state, tmp_bs = new_bz), tmp_bs = new_bz).mean()
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
        