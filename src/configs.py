k_values = [1, 5, 10, 25, 50, 100]
wd_values = [0.00005, 0.0001, 0.001, 0.01, 0.1]
bs_values = [64, 128, 256, 512]
lr_values = [0.0001, 0.001, 0.01, 0.1]


# CONFIGURATIONS FOR HYPER-PARAMETER TUNING
# hyper-parameter search configuration for the vanilla MF model
SWEEP_CONFIG_MF = {
    'method': "bayes",
    'metric': {'goal': 'maximize', 'name': 'auc'},
    'parameters': {
        'k': {"values": k_values},
        'lr': {"values": lr_values},
        'wd': {"values": wd_values},
        'tr_batch_size': {"values": bs_values}
    }
}
