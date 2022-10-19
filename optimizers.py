import torch

def get_optimizer(opt_dict, parameter):

    name = opt_dict.get("name")
    lr = opt_dict.get("lr", 1e-2)

    if name == "adam":
        opt = torch.optim.Adam(parameter)
    
    elif name == 'sgd':
        opt = torch.optim.SGD(parameter, lr=lr)

    elif name == 'sgd_momentum':
        opt = torch.optim.SGD(parameter, lr=lr, momentum=0.9)

    elif name == 'sgd_momentum_wdecay':
        decay = opt_dict.get("regularization", 0.001)
        opt = torch.optim.SGD(parameter, lr=lr, momentum=0.9, weight_decay=decay)

    return opt