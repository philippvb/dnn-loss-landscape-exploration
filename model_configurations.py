EXP_GROUPS = {
    "mobileNetV2_baseline":{"dataset":"cifar10",
            "model":"mobileNetV2",
            "loss_func": {"name":"softmax_loss", "distance":False, "factor":1, "width":1e-3, "step": 100, "multiple": False},
            "optimizer":{"name":"sgd_momentum_wdecay", "lr":1e-2, "regularization":0.001, "scheduler": "None", "step": 50, "cycle":150},
            "acc_func":{"name":"softmax_accuracy"},
            "batch_size":128,
            "max_epochs":750},

    "mobileNetV2_multiple_f10":{"dataset":"cifar10",
            "model":"mobileNetV2",
            "loss_func": {"name":"softmax_loss", "distance":True, "factor":10, "width":1e-3, "step": 150, "multiple": True},
            "optimizer":{"name":"sgd_momentum_wdecay", "lr":1e-2, "regularization":0.001, "scheduler": "None", "step": 50, "cycle":150},
            "acc_func":{"name":"softmax_accuracy"},
            "batch_size":128,
            "max_epochs":600},

    "mobileNetV2_ensemble_distance":{"dataset":"cifar10",
            "model":"mobileNetV2",
            "loss_func": {"name":"softmax_loss", "distance":True, "factor":1, "width":1e-3, "step": 150, "multiple": True},
            "optimizer":{"name":"sgd_momentum_wdecay", "lr":1e-2, "regularization":0.001, "scheduler": "cosine", "step": 50, "cycle":150},
            "acc_func":{"name":"softmax_accuracy"},
            "batch_size":128,
            "max_epochs":750}
    
}





