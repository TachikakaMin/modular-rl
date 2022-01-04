import torch
import torch.nn as nn
from dense import DenseModel
from rssm_util import get_parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WorldModel(nn.Module):
    def __init__(
        self,
        encode_size,
        args
    ):
        nn.Module.__init__(self)
        self.args = args
        self.lr = self.args.lr
        self.encode_size = encode_size
        self.state_size = self.args.limb_obs_size*self.args.max_num_limbs
        self.action_size = self.args.max_num_limbs
        self.RSSM = nn.GRUCell(self.action_size, self.encode_size).to(device)
        self.ObsEncoder = self._build_ObsEncoder(self.args.obs_encoder).to(device)
        self.ObsDecoder = self._build_ObsDecoder(self.args.obs_decoder).to(device)
        self.RSSM_forward_reward_head = self._build_forward_reward(self.args.forward_reward_args).to(device)
        self.Done_head = self._build_Done_head(self.args.done_head).to(device)
        tmp = [self.ObsEncoder, self.ObsDecoder, self.RSSM_forward_reward_head, self.rssm]
        self.model_opt = torch.optim.Adam(get_parameters(tmp), lr=self.lr)
        
        self.horizon = self.args.horizon
        self.reward_discont = 0.9
        
        self.var_networks = self._build_var_network(self.args.var_disag)
        self.var_networks = [head.to(device) for head in self.var_networks]
        self.var_opts = [torch.optim.Adam(head.parameters(), lr=self.lr) for head in self.var_networks]
        
        

    def train(self, log_var, iterations_list, batch_size, replay_buffer_list, envs_train_names):
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

                x, y, u, r, d = replay_buffer.sample_seq_len(batch_size, self.horizon)
                states = torch.FloatTensor(x).to(device)
                state = states[0]
                next_states = torch.FloatTensor(y).to(device)
                next_state = next_states[0]
                actions = torch.FloatTensor(u).to(device)
                action = actions[0]
                real_rewards = torch.FloatTensor(r).to(device)
                real_reward = real_rewards[0]
                dones = torch.FloatTensor(1 - d).to(device)
                done = dones[0]


                

                # data init
                batch_size = state.shape[0]
                state_x = torch.zeros(batch_size, self.state_size).to(device)
                state_x[:, :state.shape[1]] = state
                next_state_x = torch.zeros(batch_size, self.state_size).to(device)
                next_state_x[:, :next_state.shape[1]] = next_state




                # -----------------------------------------------
                # train encoder decoder
                embed_state = self.ObsEncoder(state_x)
                recover_state = self.ObsDecoder(embed_state)
                recover_state = recover_state[:, :state.shape[1]]
                cer = nn.SmoothL1Loss()
                loss = cer(recover_state, state)
                writer.add_scalar('{}_wm_ED_loss'.format(env_name), loss.item(), timestep)
                model_loss += loss
                # -----------------------------------------------


                # train disagreement
                with torch.no_grad():
                    embed_state = self.ObsEncoder(state_x)
                    next_embed_state = self.ObsEncoder(next_state_x)
                
                
                bz = action.shape[0]
                prev_action_x = torch.zeros(bz, self.action_size).to(device) # (bz, total_action_size)
                prev_action_x[:, :action.shape[1]] = action
                state_action_embed = torch.cat([embed_state*done, prev_action_x],dim=-1)
                for i, head in enumerate(self.var_networks):
                    pred = head(state_action_embed)
                    loss = cer(pred, next_embed_state)
                    writer.add_scalar('{}_disagree_head_{}_loss'.format(env_name, str(i)), loss.item(), timestep)
                    self.var_opts[i].zero_grad()
                    loss.backward()
                    self.var_opts[i].step()

                # train imagenation
                cer = nn.L1Loss()
                embed_state2 = self.ObsEncoder(state_x)
                self.rollout_observation(actions, dones, rssm_state)
                _, next_rssm_state = self.rssm_imagine(action, embed_state2, done)
                next_state_imagine = self.ObsDecoder(next_rssm_state)
                next_state_imagine = next_state_imagine[:, :next_state.shape[1]]
                
                loss = cer(next_state_imagine, next_state)
                writer.add_scalar('{}_wm_rnn_loss'.format(env_name), loss.item(), timestep)
                model_loss += loss

                image_done = self.Done_head(next_state_imagine)
                loss = cer(image_done, done)
                writer.add_scalar('{}_wm_done_loss'.format(env_name), loss.item(), timestep)
                model_loss += loss

                # train reward part
                cer = nn.MSELoss()
                state2_action = torch.cat([embed_state, next_embed_state], dim=-1)
                state2_action = torch.cat([state2_action, prev_action_x], dim=-1)
                pred_reward = self.RSSM_forward_reward_head(state2_action)
                loss = cer(pred_reward, real_reward)
                writer.add_scalar('{}_wm_reward_head_loss'.format(env_name), loss.item(), timestep)
                model_loss += loss

                self.model_opt.zero_grad()
                model_loss.backward()
                self.model_opt.step()


    def _build_forward_reward(self, forward_reward_args):
        input_size = self.encode_size*2 + self.action_size
        output_size = 1
        return DenseModel(input_size, output_size,  forward_reward_args)

    def _build_ObsEncoder(self, ObsEncoder_args):
        input_size = self.state_size # input = state_size
        output_size = self.encode_size # output = embed_state_size
        return DenseModel(input_size, output_size,  ObsEncoder_args)

    def _build_ObsDecoder(self, ObsDecoder_args):
        input_size = self.encode_size  # input = embed_state_size
        output_size = self.state_size # output = state_size
        return DenseModel(input_size, output_size,  ObsDecoder_args)

    def _build_var_network(self, var_args):
        input_size = self.action_size + self.encode_size  # input = embed_state + action
        output_size = self.encode_size # output = next_embed_state
        return [DenseModel(input_size, output_size, var_args, 
                # 0, [0, (i+1)/50]).to(device)
                0, [0, 1]).to(device) 
                    for i in range(10)]
        
    def _build_Done_head(self, Done_head_args):
        input_size = self.encode_size  # input = embed_state
        output_size = 1 # output = is done
        return DenseModel(input_size, output_size,  Done_head_args)

    def rssm_imagine(self, prev_action, prev_embed_state, nonterms, useDis = True):

        # make simple action to total action size
        bz = prev_action.shape[0]
        prev_action_x = torch.zeros(bz, self.action_size) # (bz, total_action_size)
        prev_action_x[:, :prev_action.shape[1]] = prev_action
        prev_action_x = prev_action_x.to(device)
        state_action_embed = torch.cat([prev_embed_state*nonterms, prev_action_x],dim=-1)

        if useDis:
            disag_reward = [head(state_action_embed) for head in self.var_networks]
            disag_reward = torch.stack(disag_reward)
            disag_reward = disag_reward.var(0).mean(-1)
            disag_reward = disag_reward[:,None]
            reward = disag_reward
        
        next_embed_state = self.RSSM(prev_action_x, prev_embed_state*nonterms)

        if not useDis:
            state2_action = torch.cat([prev_embed_state*nonterms , next_embed_state], dim=-1)
            state2_action = torch.cat([state2_action, prev_action_x], dim=-1)
            reward = self.rssm_forward_reward_head(state2_action)

        return reward, next_embed_state
    
    def rollout_imagination(self, actor:nn.Module, state, useDis = True):
        batch_size = state.shape[0]
        state_x = torch.zeros(batch_size, self.state_size).to(device)
        state_x[:, :state.shape[1]] = state
        state_x = state_x.to(device)
        prev_embed_state = self.ObsEncoder(state_x)
        rssm_state = prev_embed_state

        next_rssm_states = []
        disag_reward = []
        action_rssm = []
        reward = 0

        obs_decode = self.ObsDecoder(rssm_state)
        obs_decode = obs_decode[:, :state.shape[1]]
        for t in range(self.horizon):
            action = actor(obs_decode)
            disag, rssm_state = self.rssm_imagine(action, rssm_state, True, useDis)
            
            obs_decode = self.ObsDecoder(rssm_state)
            obs_decode = obs_decode[:, :state.shape[1]]
            next_rssm_states.append(obs_decode)
            disag_reward.append(disag*(self.reward_discont**t))
            action_rssm.append(action)
            reward += disag*(self.reward_discont**t)

        return next_rssm_states, disag_reward, action_rssm, reward

    def rollout_observation(self, actions, nonterms, rssm_state ,useDis = True):
        next_embed_states = []
        for t in range(self.horizon):
            prev_action = actions[t]*nonterms[t]
            rssm_state = self.rssm_imagine(prev_action, rssm_state, nonterms[t], useDis)
            next_embed_states.append(rssm_state)
        next_embed_states = torch.stack(next_embed_states, dim=0)
        return next_embed_states
