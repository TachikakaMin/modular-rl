import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import os 
from dataclasses import dataclass, field
from .module import get_parameters, FreezeParameters
from .algorithm import compute_return


from .dense import DenseModel
from .pixel import ObsDecoder, ObsEncoder, ConvDecoder
from .rssm import RSSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class WorldModel(object):
    def __init__(
        self,
        encode_size,
        args
    ):
        self.device = device
        self.args = args
        self.action_size = self.args.max_num_limbs
        self.state_size = self.args.limb_obs_size*self.args.max_num_limbs
        self.kl_info = {'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0}
        self.seq_len = args.seq_len
        self.batch_size = None
        self.discount = 0.99
        self.lambda_ = 0.95
        self.horizon = args.horizon
        self.loss_scale = {'kl':0.1, 'reward':1.0, 'discount':5.0}
        self.grad_clip_norm = 100
        self.rssm_info = {'deter_size':200, 'stoch_size':20, 'class_size':20, 'category_size':20, 'min_std':0.1}
        self.reward_info =  {'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU}
        self.discount_info =  {'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True}
        self.obs_encoder_info =  {'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':3, 'depth':16}
        self.obs_decoder_info =  {'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':3, 'depth':16}
        self.var_disag_info = {'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':3, 'depth':16}
        self.lr =  {'model':2e-4, 'actor':4e-5, 'critic':1e-4}
        self.embedding_size = 200
        self.rssm_node_size = 200
        
        self._model_initialize()
        self._optim_initialize()

    def train(self, log_var, iterations_list, batch_size, replay_buffer_list, envs_train_names):
        self.batch_size = batch_size
        per_morph_iter = sum(iterations_list) // len(envs_train_names)
        for i, env_name in enumerate(envs_train_names):
            log_var["env_name"] = env_name
            log_var["index"] = i
            replay_buffer = replay_buffer_list[env_name]
            model_loss = 0
            for it in range(per_morph_iter):
                env_index = log_var["index"]
                timestep = log_var["total_train_timestep_list"][env_index]
                log_var["total_train_timestep_list"][env_index] = timestep + 1
                writer = log_var["writer"]
                env_name = log_var["env_name"]

                x, y, u, r, d, imgs = replay_buffer.sample_seq_len(batch_size, self.horizon + 1)
                states = torch.FloatTensor(x).to(device)
                states = torch.permute(states, (1, 0, 2))
                obs = states[1:]
                obs_tmp = torch.zeros(self.horizon , batch_size, self.state_size).to(device)
                obs_tmp[:, :, :obs.shape[2]] = obs
                obs = obs_tmp

                actions = torch.FloatTensor(u).to(device)
                actions = torch.permute(actions, (1, 0, 2))
                acts = actions[:-1]
                acts_tmp = torch.zeros(self.horizon, batch_size, self.action_size).to(device)
                acts_tmp[:, :, :acts.shape[2]] = acts
                acts = acts_tmp
                
                real_rewards = torch.FloatTensor(r).to(device)
                real_rewards = torch.permute(real_rewards, (1, 0, 2))
                rewards = real_rewards[:-1]

                dones = torch.FloatTensor(1 - d).to(device)
                dones = torch.permute(dones, (1, 0, 2)) # (horizon, bz, size)
                nonterms = dones[:-1]

                model_loss, kl_loss, obs_loss, reward_loss, disag_losses,  pcont_loss, prior_dist, post_dist, posterior = self.representation_loss(obs, acts, rewards, nonterms)
                writer.add_scalar('{}_wm_model_loss'.format(env_name), model_loss.item(), timestep)
                writer.add_scalar('{}_wm_kl_loss'.format(env_name), kl_loss.item(), timestep)
                writer.add_scalar('{}_wm_obs_loss'.format(env_name), obs_loss.item(), timestep)
                writer.add_scalar('{}_wm_reward_loss'.format(env_name), reward_loss.item(), timestep)
                writer.add_scalar('{}_wm_reward_loss'.format(env_name), reward_loss.item(), timestep)
                writer.add_scalar('{}_wm_reward_loss'.format(env_name), reward_loss.item(), timestep)
                # for k in range(10):
                #     writer.add_scalar('{}_wm_disag_head_{}_loss'.format(env_name, k), disag_losses[k].item(), timestep)
                writer.add_scalar('{}_wm_pcont_loss'.format(env_name), pcont_loss.item(), timestep)
                 
                self.model_optimizer.zero_grad()
                model_loss.backward()
                grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
                self.model_optimizer.step()

                # self.var_optimizer.zero_grad()
                # disag_loss = sum(disag_losses)
                # disag_loss.backward()
                # grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.var_networks), self.grad_clip_norm)
                # self.var_optimizer.step()


    def get_disag_reward(self, action_state_embeds):
        disag_dist = [head(action_state_embeds).sample() for head in self.var_networks]
        disag = torch.stack(disag_dist, dim=0)
        disag = disag.std(0).mean(-1)
        return disag

    def representation_loss(self, obs, actions, rewards, nonterms):

        embed = self.ObsEncoder(obs)                                         #t to t+seq_len   
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)   
        prior, posterior, action_state_embeds = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len   
        obs_dist = self.ObsDecoder(post_modelstate[:-1])                     #t to t+seq_len-1  
        reward_dist = self.RewardDecoder(post_modelstate[:-1])               #t to t+seq_len-1  
        pcont_dist = self.DiscountModel(post_modelstate[:-1])                #t to t+seq_len-1   
        # disag_dist = [head(action_state_embeds.detach()) for head in self.var_networks]

        disag_losses = [0]
        # disag_losses = self._disag_loss(disag_dist, post_modelstate.detach())
        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = self.loss_scale['kl'] * div + reward_loss + obs_loss + self.loss_scale['discount']*pcont_loss
        return model_loss, div, obs_loss, reward_loss, disag_losses, pcont_loss, prior_dist, post_dist, posterior
        
    def _disag_loss(self, disag_dists, post_modelstate):
        disag_loss = []
        for disag_dist in disag_dists:
            disag_loss.append(-torch.mean(disag_dist.log_prob(post_modelstate)))
        return disag_loss

    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def _model_initialize(self):
        obs_size = self.state_size
        action_size = self.action_size
        deter_size = self.rssm_info['deter_size']
        stoch_size = self.rssm_info['stoch_size']
        embedding_size = self.embedding_size
        rssm_node_size = self.rssm_node_size
        modelstate_size = stoch_size + deter_size 
    
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, self.rssm_info).to(self.device)
        self.RewardDecoder = DenseModel(1, modelstate_size, self.reward_info).to(self.device)
        
        if self.discount_info['use']:
            self.DiscountModel = DenseModel(1, modelstate_size, self.discount_info).to(self.device)
        
        self.ObsEncoder = DenseModel(embedding_size, obs_size, self.obs_encoder_info).to(self.device)
        self.ObsDecoder = DenseModel(obs_size, modelstate_size, self.obs_decoder_info).to(self.device)
        self.var_networks = [DenseModel(modelstate_size, deter_size, self.var_disag_info).to(self.device) for i in range(10)]
        self.var_networks = [head.to(device) for head in self.var_networks]

        from torchsummary import summary
        
        self.Obs2image = ConvDecoder(obs_size, 32, nn.ELU, self.args.img_shape).to(self.device)
        # summary(self.Obs2image, (16, obs_size))


    def _optim_initialize(self):
        model_lr = self.lr['model']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.var_optimizer = optim.Adam(get_parameters(self.var_networks), model_lr)
        self.obs2img_optimizer = optim.Adam(get_parameters([self.Obs2image]), model_lr)

    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount['use']:
            print('\n Discount decoder: \n', self.DiscountModel)
        