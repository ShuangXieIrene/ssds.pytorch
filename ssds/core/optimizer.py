import torch.optim as optim
from torch.optim import lr_scheduler


class InvertedExponentialLR(lr_scheduler._LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.

    :meta private:
    """

    def __init__(self, optimizer, end_lr, num_iter=100, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def trainable_param(model, trainable_scope):
    r""" Return the trainable parameters for the optimizers by :attr:`cfg.TRAIN.TRAINABLE_SCOPE`

    If the module in trainable scope, then train this module's parameters

    When :

    * cfg.TRAIN.TRAINABLE_SCOPE = ""
        All the parameters in the model are used to train
    * cfg.TRAIN.TRAINABLE_SCOPE = "a,b,c.d"
        Only the the parameters in the a, b and c.d are used to train
    * cfg.TRAIN.TRAINABLE_SCOPE = "a;b,c.d"
        Only the the parameters in the a, b and c.d are used to train. module a and model b&c.d can be assigned to different learning rate (differential learning rate)

    Args:
        model: the ssds model for training
        trainable_scope (str): the scope for the trainable parameter in the given ssds model, which is defined in the cfg.TRAIN.TRAINABLE_SCOPE
    """
    trainable_param = []

    if trainable_scope == "":
        for param in model.parameters():
            param.requires_grad = True
        trainable_param.append(model.parameters())
    else:
        for param in model.parameters():
            param.requires_grad = False

        for train_scope in trainable_scope.split(";"):
            param_temp = []
            for module in train_scope.split(","):
                submodule = module.split(".")
                tmp_model = model
                for subm in submodule:
                    if hasattr(tmp_model, subm):
                        tmp_model = getattr(tmp_model, subm)
                    else:
                        raise ValueError(module + " is not in the model")
                for param in tmp_model.parameters():
                    param.requires_grad = True
                param_temp.extend(tmp_model.parameters())
            trainable_param.append(param_temp)
    return trainable_param


def configure_optimizer(trainable_param, cfg):
    r""" Return the optimizer for the trainable parameters

    Basically, it returns the optimizer defined by :attr:`cfg.TRAIN.OPTIMIZER.OPTIMIZER`. The learning rate for the optimizer is defined by :attr:`cfg.TRAIN.OPTIMIZER.LEARNING_RATE`
    and :attr:`cfg.TRAIN.OPTIMIZER.DIFFERENTIAL_LEARNING_RATE`. Some other parameters are also defined in :attr:`cfg.TRAIN.OPTIMIZER`.

    Currently, there are 4 popular optimizers supported: sgd, rmsprop, adam and amsgrad.

    TODO: directly fetch the optimizer by getattr(optim, cfg.OPTIMIZER) and send the the relative parameter by dict.

    Args:
        trainable_param: the trainable parameter in the given ssds model, check :meth:`trainable_param` for more details.
        cfg: the config dict, which is defined in :attr:`cfg.TRAIN.OPTIMIZER`. 
    """


    if len(cfg.DIFFERENTIAL_LEARNING_RATE) == 0 or len(trainable_param) == 1:
        trainable_param = trainable_param[0]
    else:
        assert len(cfg.DIFFERENTIAL_LEARNING_RATE) == len(trainable_param)
        trainable_param = [
            {"params": _param, "lr": _lr}
            for _param, _lr in zip(trainable_param, cfg.DIFFERENTIAL_LEARNING_RATE)
        ]

    if cfg.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            trainable_param,
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            trainable_param,
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            alpha=cfg.MOMENTUM_2,
            eps=cfg.EPS,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            trainable_param,
            lr=cfg.LEARNING_RATE,
            betas=(cfg.MOMENTUM, cfg.MOMENTUM_2),
            weight_decay=cfg.WEIGHT_DECAY,
        )
    elif cfg.OPTIMIZER == "amsgrad":
        optimizer = optim.Adam(
            trainable_param,
            lr=cfg.LEARNING_RATE,
            betas=(cfg.MOMENTUM, cfg.MOMENTUM_2),
            weight_decay=cfg.WEIGHT_DECAY,
            amsgrad=True,
        )
    else:
        AssertionError("optimizer can not be recognized")
    return optimizer


def configure_lr_scheduler(optimizer, cfg):
    r""" Return the learning rate scheduler for the trainable parameters

    Basically, it returns the learning rate scheduler defined by :attr:`cfg.TRAIN.LR_SCHEDULER.SCHEDULER`. 
    Some parameters for the learning rate scheduler are also defined in :attr:`cfg.TRAIN.LR_SCHEDULER`.

    Currently, there are 4 popular learning rate scheduler supported: step, multi_step, exponential and sgdr.

    TODO: directly fetch the optimizer by getattr(lr_scheduler, cfg.SCHEDULER) and send the the relative parameter by dict.

    Args:
        optimizer: the optimizer in the given ssds model, check :meth:`configure_optimizer` for more details.
        cfg: the config dict, which is defined in :attr:`cfg.TRAIN.LR_SCHEDULER`. 
    """
    if cfg.SCHEDULER == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA
        )
    elif cfg.SCHEDULER == "multi_step":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA
        )
    elif cfg.SCHEDULER == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == "inverted_exponential":
        scheduler = InvertedExponentialLR(optimizer, end_lr=cfg.LR_MIN)
    elif cfg.SCHEDULER == "sgdr":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2, T_mult=2, eta_min=cfg.LR_MIN
        )
    else:
        AssertionError("scheduler can not be recognized.")
    return scheduler
