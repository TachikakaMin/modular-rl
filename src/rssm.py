import torch
import torch.nn as nn
from rssm_util import RSSMUtils, RSSMContState
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RSSM(nn.Module, RSSMUtils):
    def __init__(
        self,
        action_size,
        rssm_node_size,
        embedding_size,
        device,
        rssm_type,
        info,
        act_fn=nn.ELU,  
    ):
        nn.Module.__init__(self)
        RSSMUtils.__init__(self, rssm_type=rssm_type, info=info)
        self.device = device
        self.action_size = action_size
        self.node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.act_fn = act_fn
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()
        self.reward_discont = 0.9
    
    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action 
        and embed it to deter size for rnn input
        """
        fc_embed_state_action = [nn.Linear(self.stoch_size + self.action_size, self.deter_size)]
        fc_embed_state_action += [self.act_fn()]
        return nn.Sequential(*fc_embed_state_action)
    
    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state 
        and output prior over stochastic state
        """
        temporal_prior = [nn.Linear(self.deter_size, self.node_size)]
        temporal_prior += [self.act_fn()]
        temporal_prior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_prior)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state 
        and output posterior over stochastic states
        """
        temporal_posterior = [nn.Linear(self.deter_size + self.embedding_size, self.node_size)]
        temporal_posterior += [self.act_fn()]
        temporal_posterior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_posterior)
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms, var_networks):
        prev_action_x = torch.zeros(prev_action.shape[0], self.action_size)
        prev_action_x[:, :prev_action.shape[1]] = prev_action
        prev_action_x = prev_action_x.to(device)
        state_action_embed = self.fc_embed_state_action(
                torch.cat([prev_rssm_state.stoch*nonterms, prev_action_x],dim=-1)
            )
        preds_disag = [head(state_action_embed) for head in var_networks]
        preds_disag = torch.stack(preds_disag)
        disag = preds_disag.std(0).mean(-1)
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter*nonterms)
        prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
        stats = {'mean':prior_mean, 'std':prior_std}
        prior_stoch_state, std = self.get_stoch_state(stats)
        prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state, disag, preds_disag

    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state, var_networks, ObsDecoder):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        rssm_reward = []
        action_rssm = []
        for t in range(horizon):
            state = (self.get_model_state(rssm_state)).detach()
            obs_decode = ObsDecoder(state)
            tot_num = obs_decode.shape[1]//self.action_size*actor.num_limbs
            action = actor(obs_decode[:,:tot_num])
            rssm_state, disag, _ = self.rssm_imagine(action, rssm_state, True, var_networks)
            rssm_reward.append(disag)
            action_rssm.append(action_rssm)
            next_rssm_states.append(rssm_state)
            
        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        return next_rssm_states, rssm_reward, action_rssm

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state, var_networks):
        prior_rssm_state, _, preds_disag = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm, var_networks)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
        stats = {'mean':posterior_mean, 'std':posterior_std}
        posterior_stoch_state, std = self.get_stoch_state(stats)
        posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state, preds_disag

    def rollout_observation(self, 
                            seq_len:int, 
                            obs_embed: torch.Tensor, 
                            action: torch.Tensor, 
                            nonterms: torch.Tensor, 
                            prev_rssm_state, 
                            var_networks: list):
        priors = []
        posteriors = []
        preds_disags = []
        for t in range(seq_len-1):
            prev_action = action[:,t]*nonterms[:,t]
            prior_rssm_state, posterior_rssm_state, preds_disag = self.rssm_observe(obs_embed[:,t], prev_action, nonterms[:,t], prev_rssm_state, var_networks)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            preds_disags.append(preds_disag)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=1)
        post = self.rssm_stack_states(posteriors, dim=1)
        preds_disags = torch.stack(preds_disags)
        return prior, post, preds_disags
        