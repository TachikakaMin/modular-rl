import torch
import torch.nn as nn
from .rssm_util import RSSMUtils, RSSMContState, RSSMDiscState
from .module import get_parameters, FreezeParameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
class RSSM(nn.Module, RSSMUtils):
    def __init__(
        self,
        action_size,
        rssm_node_size,
        embedding_size,
        device,
        info,
        act_fn=nn.ELU,  
    ):
        nn.Module.__init__(self)
        RSSMUtils.__init__(self, info=info)
        self.device = device
        self.action_size = action_size
        self.node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.act_fn = act_fn
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()
    
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
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        state_action_embed = self.fc_embed_state_action(torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1))
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter*nonterms)
        prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
        stats = {'mean':prior_mean, 'std':prior_std}
        prior_stoch_state, std = self.get_stoch_state(stats)
        prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state, state_action_embed

    def rollout_imagination(self, horizon:int, actor:nn.Module, modular_state_size, prev_rssm_state, ObsDecoder):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        actions = []
        state_action_embeds = []
        # imag_log_probs = []
        for t in range(horizon):
            # action, action_dist = actor((self.get_model_state(rssm_state)).detach())
            input_state = (self.get_model_state(rssm_state)).detach()
            with FreezeParameters([ObsDecoder]):
                input_state = ObsDecoder(input_state)   
            input_state = input_state.sample()[:, :modular_state_size]
            action = actor(input_state, tmp_bs=input_state.shape[0])
            tmp = torch.zeros(action.shape[0], self.action_size).to(device)
            tmp[:, :action.shape[1]] = action
            action = tmp
            rssm_state, state_action_embed = self.rssm_imagine(action, rssm_state)
            state_action_embeds.append(state_action_embed)
            next_rssm_states.append(rssm_state)
            actions.append(action)
            # imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        actions = torch.stack(actions, dim=0)
        state_action_embeds = torch.stack(state_action_embeds, dim=0)
        # imag_log_probs = torch.stack(imag_log_probs, dim=0)
        # return next_rssm_states, imag_log_probs, action_entropy
        return next_rssm_states, actions, state_action_embeds

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        prior_rssm_state, state_action_embed  = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        
        posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
        stats = {'mean':posterior_mean, 'std':posterior_std}
        posterior_stoch_state, std = self.get_stoch_state(stats)
        posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state, state_action_embed

    def rollout_observation(self, seq_len:int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state):
        priors = []
        posteriors = []
        action_state_embeds = []
        for t in range(seq_len):
            prev_action = action[t]*nonterms[t]
            prior_rssm_state, posterior_rssm_state, state_action_embed = self.rssm_observe(obs_embed[t], prev_action, nonterms[t], prev_rssm_state)
            action_state_embeds.append(state_action_embed)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=0)
        post = self.rssm_stack_states(posteriors, dim=0)
        action_state_embeds = torch.stack(action_state_embeds, dim=0)
        return prior, post, action_state_embeds
        