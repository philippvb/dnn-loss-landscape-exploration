Queue = {
    "cifar10_1":{"dataset":"cifar10",
            "model":"resnet34",
            "loss_func": {"name":"softmax_loss", "distance":False, "factor":1, "width":1},
            "optimizer":{"name":"sgd_momentum_wdecay", "lr":1e-3, "regularization":0.01},
            "acc_func":{"name":"softmax_accuracy"},
            "batch_size":128,
            "max_epochs":150},

        "cifar10_2":{"dataset":"cifar10",
            "model":"resnet34",
            "loss_func": {"name":"softmax_loss", "distance":True, "factor":1, "width":1e-1},
            "optimizer":{"name":"sgd_momentum_wdecay", "lr":1e-3, "regularization":0.01},
            "acc_func":{"name":"softmax_accuracy"},
            "batch_size":128,
            "max_epochs":150},

        "cifar10_3":{"dataset":"cifar10",
            "model":"resnet34",
            "loss_func": {"name":"softmax_loss", "distance":True, "factor":1, "width":1e-2},
            "optimizer":{"name":"sgd_momentum_wdecay", "lr":1e-3, "regularization":0.01},
            "acc_func":{"name":"softmax_accuracy"},
            "batch_size":128,
            "max_epochs":150},

        "cifar10_4":{"dataset":"cifar10",
            "model":"resnet34",
            "loss_func": {"name":"softmax_loss", "distance":True, "factor":1, "width":1e-3},
            "optimizer":{"name":"sgd_momentum_wdecay", "lr":1e-3, "regularization":0.01},
            "acc_func":{"name":"softmax_accuracy"},
            "batch_size":128,
            "max_epochs":150}
}