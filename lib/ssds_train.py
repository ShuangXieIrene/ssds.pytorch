from __future__ import print_function
import numpy as np
import os
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.nms_wrapper import nms
from lib.dataset.data_augment import preproc

from lib.dataset import dataset_factory
from lib.nets import net_factory
from lib.models import model_factory
from lib.utils.config_parse import cfg


class SolverWrapper(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, base_fn=None, model_fn=None, dataset_fn=None, max_epochs=None, output_dir=None, tbdir=None, resume_checkpoint=None, gpus = None):
        if base_fn is None:
            base_fn = cfg.MODEL.BASE_FN
        base = net_factory.gen_base_fn(name=base_fn)
        if model_fn is None:
            model_fn = cfg.MODEL.MODEL_FN
        self.net = model_factory.gen_model_fn(name=model_fn)(base=base, 
                    feature_layer=cfg.MODEL.FEATURE_LAYER, layer_depth=cfg.MODEL.LAYER_DEPTH, mbox=cfg.MODEL.MBOX, num_classes=21)
        if dataset_fn is None:
            dataset_fn = cfg.DATASET_FN
            self.dataset = dataset_factory.gen_dataset_fn(name=dataset_fn)(cfg.DATA_DIR, cfg.TRAIN_SETS, preproc(cfg.MODEL.IMAGE_SIZE, cfg.MODEL.PIXEL_MEANS, cfg.PROB))
        if max_epochs is None:
            max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.max_epochs = max_epochs
        if output_dir is None:
            output_dir = cfg.EXP_DIR
        self.output_dir = output_dir
        if tbdir is None:
            tbdir = cfg.LOG_DIR
        self.tbdir = tbdir
        self.tbvaldir = tbdir + '_val'
        if resume_checkpoint is None:
            resume_checkpoint=cfg.RESUME_CHECKPOINT
        self.resume_checkpoint = resume_checkpoint
        self.checkpoint_prefix = '{}_{}'.format(base_fn, model_fn)
        self.gpus = gpus

        feature_maps = self.net._forward_features_size(cfg.MODEL.IMAGE_SIZE)
        priorbox = PriorBox(image_size=cfg.MODEL.IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=cfg.MODEL.PRIOR_BOX.ASPECT_RATIOS, 
                    scale=cfg.MODEL.PRIOR_BOX.SIZES, archor_stride=cfg.MODEL.PRIOR_BOX.STEPS, clip=cfg.MODEL.PRIOR_BOX.CLIP)
        self.priors = Variable(priorbox.forward(), volatile=True)

        # print(self.net)
        
    
    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.net.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('\nWrote snapshot to: {:s}'.format(filename))
        
        # TODO: write relative cfg under the same page
    
    def restore_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '':
            return False
        print('Restoring checkpoint from {:s}'.format(resume_checkpoint))
        return self.net.load_weights(resume_checkpoint, cfg.TRAIN.RESUME_SCOPE)

    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        line = lineList[-1]
        start_epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
        resume_checkpoint = line[line.find(':') + 2:-1]
        return start_epoch, resume_checkpoint

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    def initialize(self):
        # Fresh train directly from ImageNet weights
        if self.resume_checkpoint:
            print('Loading initial model weights from {:s}'.format(self.resume_checkpoint))
            self.restore_checkpoint(self.resume_checkpoint)        
        else:
        # TODO: ADD INIT ways
            self.net.extras.apply(self.weights_init)
            self.net.loc.apply(self.weights_init)
            self.net.conf.apply(self.weights_init)
        start_epoch = 0
        return start_epoch
    
    def trainable_param(self, trainable_scope):
        for param in self.net.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.net, module):
                # print(getattr(self.net, module))
                for param in getattr(self.net, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.net, module).parameters())
                    
        return trainable_param

    def train_model(self):
        previous = self.find_previous()
        if previous:
            start_epoch = previous[0]
            self.restore_checkpoint(previous[1])
        else:
            start_epoch = self.initialize()
        
        # Whether gpu is avaiable or not
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.net.cuda()
            if self.gpus is not None: torch.nn.DataParallel(self.net, device_ids=[int(i) for i in self.gpus.strip().split(',')])
            # else: torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        self.net.train()

        trainable_param = self.trainable_param(cfg.TRAIN.TRAINABLE_SCOPE)
        optimizer = optim.SGD(trainable_param, lr=cfg.TRAIN.LEARNING_RATE,
                        momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.STEPSIZE, gamma=cfg.TRAIN.GAMMA)
        # load the relative hyperpremeter
        criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, use_gpu)


        # get dataset size
        epoch_size = len(self.dataset) // cfg.TRAIN.BATCH_SIZE
        data_loader = data.DataLoader(self.dataset, cfg.TRAIN.BATCH_SIZE, num_workers=4,
                                  shuffle=True, collate_fn=dataset_factory.detection_collate, pin_memory=True)
        _t = Timer()
        for epoch in iter(range(start_epoch+1, self.max_epochs)):
            #learning rate
            exp_lr_scheduler.step(epoch)
            #batch data
            batch_iterator = iter(data_loader)
            loc_loss = 0
            conf_loss = 0
            for iteration in iter(range((epoch_size))):
                images, targets = next(batch_iterator)
                if use_gpu:
                    images = Variable(images.cuda())
                    targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(anno, volatile=True) for anno in targets]

                _t.tic()
                # forward
                out = self.net(images, is_train=True)

                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets, self.priors)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()

                time = _t.toc()
                loc_loss += loss_l.data[0]
                conf_loss += loss_c.data[0]

                self.add_summary(epoch, iteration, epoch_size, loss_l.data[0], loss_c.data[0], time, optimizer)

                if iteration % 10 == 9:
                    break
            if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                self.save_checkpoints(epoch)

        
    def add_summary(self, epoch, iters, epoch_size, loc_loss, conf_loss, time, optim):
        if iters == 0:
            sys.stdout.write('\n')
        lr = optim.param_groups[0]['lr']
        log = '\rEpoch {epoch:d}: {iters:d}/{epoch_size:d} in {time:.2f}s [{prograss}] || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\r'.format(epoch=epoch, lr=lr,
                prograss='#'*int(round(10*iters/epoch_size)) + '-'*int(round(10*(1-iters/epoch_size))), iters=iters, epoch_size=epoch_size, 
                time=time, loc_loss=loc_loss, conf_loss=conf_loss)
        sys.stdout.write(log)
        sys.stdout.flush()
        return True

    def predict(self, img):
        return True



def train_model(base_fn=None, model_fn=None, max_epochs=None, output_dir=None, tbdir=None, resume_checkpoint=None, gpus=None, prior_box=None):
    sw = SolverWrapper(base_fn, model_fn, max_epochs, output_dir, tbdir, resume_checkpoint, gpus, prior_box)
    sw.train_model()
    return True