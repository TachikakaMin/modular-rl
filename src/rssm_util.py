from collections import namedtuple
import torch.distributions as td
import torch
import torch.nn.functional as F
from typing import Union
from typing import Iterable

RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])  

RSSMState = RSSMContState

def get_parameters(modules: Iterable[torch.nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
    def __init__(self, modules: Iterable[torch.nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

class RSSMUtils(object):
    '''utility functions for dealing with rssm states'''
    def __init__(self, rssm_type, info):
        self.rssm_type = rssm_type
        self.deter_size = info['deter_size']
        self.stoch_size = info['stoch_size']
        self.min_std = info['min_std']

    def rssm_seq_to_batch(self, rssm_state, seq_len):
        return RSSMContState(
            rssm_state.mean[:,:seq_len],
            rssm_state.std[:,:seq_len],
            rssm_state.stoch[:,:seq_len], 
            rssm_state.deter[:,:seq_len],
        )
        
    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len):
        return RSSMContState(
            batch_to_seq(rssm_state.mean, batch_size, seq_len),
            batch_to_seq(rssm_state.std, batch_size, seq_len),
            batch_to_seq(rssm_state.stoch, batch_size, seq_len),
            batch_to_seq(rssm_state.deter, batch_size, seq_len)
        )
        
    def get_dist(self, rssm_state):
        return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)

    def get_stoch_state(self, stats):
        mean = stats['mean']
        std = stats['std']
        std = F.softplus(std) + self.min_std
        return mean + std*torch.randn_like(mean), std

    def rssm_stack_states(self, rssm_states, dim):
        return RSSMContState(
            torch.stack([state.mean for state in rssm_states], dim=dim),
            torch.stack([state.std for state in rssm_states], dim=dim),
            torch.stack([state.stoch for state in rssm_states], dim=dim),
            torch.stack([state.deter for state in rssm_states], dim=dim),
        )

    def get_model_state(self, rssm_state):
        return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

    def rssm_detach(self, rssm_state):
        return RSSMContState(
            rssm_state.mean.detach(),
            rssm_state.std.detach(),  
            rssm_state.stoch.detach(),
            rssm_state.deter.detach()
        )

    def _init_rssm_state(self, batch_size, **kwargs):
        return RSSMContState(
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
            torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
        )
            
def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
    return seq_data

