from Utilities import *
from Network import *

from ValueNetworks import *
from PolicyNetworks import *

NETWORK_CONFIGS = {
    'mlp': {
        'mini'  : [32, 16],
        'small' : [128, 64],
        'medium' : [256, 128, 64],
        'large' : [512, 256, 128, 64],
    },
    'cnn' : {
        'ALE' : {
            'conv_layers' : [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)], # (out_channels, kernel_size, stride, padding)
            'fc_layers' : [512],
        },
    }
}

def create_value_network(input_size: int, config: dict[str, Any])->ValueNetwork:
    """
    Creates and returns a Value Network with the architecture that was specified in the configuration file.
    Args:
        input_size (int): observation dim
        config (dict[str, Any]): Configuration dictionary, as loaded from the configuration file

    Returns:
        ValueNetwork: Any derivation of Value Network, currently either ValueMLP or ValueCNN
    """

    if config['network_type'] == 'mlp':
        return ValueMLP(input_size=input_size,
                        hidden_layers=NETWORK_CONFIGS[config['network_type']][config['value_net_size']],
                        name=config['value_net_size'])
    
    elif config['network_type'] == 'cnn':
        cfg = NETWORK_CONFIGS[config['network_type']][config['policy_net_size']]
        conv_layers = cfg['conv_layers']
        fc_layers = cfg['fc_layers']
        return ValueCNN(4, 84, 84, conv_layers, fc_layers)
    
def create_policy_network(input_size: int, output_size: int, config: dict[str, Any])->PolicyNetwork:
    """Creates and returns a Policy Network with the architecture that was specified in the configuration file

    Args:
        input_size (int): Observation dim
        output_size (int): Action Space dim
        config (dict[str, Any]): Configuration dictionary, as loaded from the configuration file

    Returns:
        PolicyNetwork: Any derivation of Policy Network depending on the Policy Type (in configuration file)\\
        Currently either PolicyTypeMLP or PolicyTypeCNN, for PolicyType: Logits, GNN, GNN_K, GNN_N
    """

    if config['network_type'] == 'mlp':
        if config['policy_type'] == 'logits':
            return LogitsMLP(input_size=input_size,
                             output_size=output_size,
                             hidden_layers=NETWORK_CONFIGS[config['network_type']][config['policy_net_size']],
                             name=config['policy_net_size'])
        
        elif config['policy_type'] == 'GNN':

            # if static :
            #   do the intervals here
            #   do the mapping here
            # if in the future I decide to make intervals and mapping dynamic, the network will predict parameters that correspond to it
            # and the mapping will be done in each forward call. (NOTE: I will not, these methods do not work).
            
            interval_fn = resolve_interval_fn[config['intervals']['fn_name']]
            mapping_fn = resolve_mapping_fn[config['mapping']['fn_name']]

            intervals = interval_fn(**config['intervals']['kwargs'])
            mapping = mapping_fn(**config['mapping']['kwargs'])

            net = GNN_MLP(input_size=input_size,
                           hidden_layers=NETWORK_CONFIGS[config['network_type']][config['policy_net_size']],
                           name=config['policy_net_size'],
                           intervals=intervals,
                           mapping=mapping)
            
            with torch.no_grad():
                for param in net.network[-1].parameters():  # final layer params set to 0 to ensure 0 mean start position.
                    param.fill_(0.0)                        # this is the center of the mapping interval and will allow a good start to learn

            return net


        elif config['policy_type'] == 'GNN_K':
            # if static :
            #   do the intervals here
            #   do the mapping here
            # if in the future I decide to make intervals and mapping dynamic, the network will predict parameters that correspond to it
            # and the mapping will be done in each forward call. (NOTE: I will not, these methods do not work).
            
            interval_fn = resolve_interval_fn[config['intervals']['fn_name']]
            mapping_fn = resolve_mapping_fn[config['mapping']['fn_name']]

            intervals = interval_fn(**config['intervals']['kwargs'])
            mapping = mapping_fn(**config['mapping']['kwargs'])

            net = GNN_K_MLP(input_size=input_size,
                           hidden_layers=NETWORK_CONFIGS[config['network_type']][config['policy_net_size']],
                           name=config['policy_net_size'],
                           intervals=intervals,
                           mapping=mapping,
                           K=config['K_num_components'])
            
            with torch.no_grad():
                for param in net.network[-1].parameters():  # final layer params set to 0 to ensure 0 mean start position.
                    param.fill_(0.0)                        # this is the center of the mapping interval and will allow a good start to learn

            return net


        elif config['policy_type'] == 'GNN_N':
            return GNN_N_MLP(input_size=input_size,
                           hidden_layers=NETWORK_CONFIGS[config['network_type']][config['policy_net_size']],
                           name=config['policy_net_size'],
                           N = output_size,
                           confidence_hidden_multiplier=config['confidence_hidden_multiplier'])
        
    elif config['network_type'] == 'cnn':        # input shape is hardcoded for ALE environments only: 4x84x84 (grayscale framestack)

        cfg = NETWORK_CONFIGS[config['network_type']][config['policy_net_size']]
        conv_layers = cfg['conv_layers']
        fc_layers = cfg['fc_layers']

        if config['policy_type'] == 'logits':
            return LogitsCNN(output_size, 4, 84, 84, conv_layers, fc_layers)
        
        # elif config['policy_type'] == 'GNN':      # no image-based experiments for GNN and GNN_K methods.
        #     return GNN_CNN()
        # elif config['policy_type'] == 'GNN_K':
        #     return GNN_K_CNN()

        elif config['policy_type'] == 'GNN_N':
            return GNN_N_CNN(output_size, 4, 84, 84, conv_layers, fc_layers, noise_coeff=config['noise_coeff'])
