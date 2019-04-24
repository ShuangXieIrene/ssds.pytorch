import torch.optim as optim
from torch.optim import lr_scheduler

def trainable_param(model, trainable_scope):
    '''
    given the trainable module by cfg.TRAINABLE_SCOPE, 
    if the module in trainable scope, then train this module's parameters
    '''
    trainable_param = []

    if trainable_scope == '':
        for param in model.parameters():
            param.requires_grad = True
        trainable_param.extend(model.parameters())
    else:
        for param in model.parameters():
            param.requires_grad = False

        for module in trainable_scope.split(','):
            if hasattr(model, module):
                for param in getattr(model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(model, module).parameters())
        
    # for param in model.base.parameters():
    #     print(param.name)
    # assert False
    return trainable_param

def configure_optimizer(trainable_param, cfg):
    if cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(trainable_param, lr=cfg.LEARNING_RATE,
                    momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(trainable_param, lr=cfg.LEARNING_RATE,
                    momentum=cfg.MOMENTUM, alpha=cfg.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(trainable_param, lr=cfg.LEARNING_RATE,
                    betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), weight_decay=cfg.WEIGHT_DECAY)
    elif cfg.OPTIMIZER == 'amsgrad':
        optimizer = optim.Adam(trainable_param, lr=cfg.LEARNING_RATE,
                    betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), weight_decay=cfg.WEIGHT_DECAY, amsgrad=True)
    else:
        AssertionError('optimizer can not be recognized')
    return optimizer

def configure_lr_scheduler(optimizer, cfg):
    if cfg.SCHEDULER == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'SGDR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
    else:
        AssertionError('scheduler can not be recognized.')
    return scheduler