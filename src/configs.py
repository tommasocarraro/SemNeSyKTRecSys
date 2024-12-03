k_values = [5, 10, 25, 50]
wd_values = [0.00005, 0.0001, 0.001, 0.01]
bs_values = [64, 128, 256]
lr_values = [0.0001, 0.001, 0.01]


# CONFIGURATIONS FOR HYPER-PARAMETER TUNING
# hyper-parameter search configuration for the vanilla MF model
SWEEP_CONFIG_MF = {
    'name': "mf",
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values}
    }
}

SWEEP_CONFIG_MF_TOY = {
    'name': "mf",
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": [1, 2, 3]},
        'lr': {"values": [0.1, 0.001, 0.01]},
        'wd': {"values": [0.0001, 0.00001, 0.001]},
        'tr_batch_size': {"values": [32, 64, 128, 256, 512]}
    }
}
