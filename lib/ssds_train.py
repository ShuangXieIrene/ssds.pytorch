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
    def __init__(self, base_fn=None, model_fn=None, dataset_fn=None, max_epochs=None, output_dir=None, tbdir=None, resume_checkpoint=None, gpus = None, prior_box=None):
        if base_fn is None:
            base_fn = cfg.MODEL.BASE_FN
        self.base = net_factory.gen_base_fn(name=base_fn)
        if model_fn is None:
            model_fn = cfg.MODEL.MODEL_FN
        self.net = model_factory.gen_model_fn(name=model_fn)(base=self.base)
        if dataset_fn is None:
            dataset_fn = cfg.DATASET_FN
            self.dataset = dataset_factory.gen_dataset_fn(name=dataset_fn)(cfg.DATA_DIR, cfg.TRAIN_SETS, preproc(cfg.IMG_SIZE, cfg.RGB_MEAN, cfg.PROB))
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
        self.resume_checkpoint = resume_checkpoint
        self.checkpoint_prefix = '{}_{}'.format(base_fn, model_fn)
        self.gpus = gpus
        if prior_box is None:
            prior_box = cfg.MODEL.PRIOR_BOX
        self.priorbox = PriorBox(prior_box)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
    
    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters is None:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.net.state_dict(), filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        # TODO: add checkpoints list
        # TODO: write relative cfg filename
    
    def restore_checkpoint(self, resume_checkpoint):
        if resume_checkpoint is None:
            return False
        print('Restoring checkpoint from {:s}'.format(resume_checkpoint))
        return self.net.load_weights(resume_checkpoint)

    def find_previous(self):
        import glob
        files = os.path.join(self.output_dir, '*.pth')
        files = glob.glob(files)
        # TODO: use checkpoint list to find previous
        if len(files) == 0:
            return False
        files.sort(key=os.path.getmtime)
        return files[-1]

    def initialize(self):
        # Fresh train directly from ImageNet weights
        if self.resume_checkpoint is not None:
            print('Loading initial model weights from {:s}'.format(self.resume_checkpoint))
            self.restore_checkpoint(self.resume_checkpoint)        
        # Need to fix the variables before loading, so that the RGB weights are changed to BGR
        # For VGG16 it also changes the convolutional weights fc6 and fc7 to
        # fully connected weights

        # TODO: ADD INIT ways

        iter = 0
        return iter

    def train_model(self):
        previous = self.find_previous()
        if previous:
            start_epoch = self.restore_checkpoint(previous)
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

        optimizer = optim.SGD(self.net.parameters(), lr=cfg.TRAIN.LEARNING_RATE,
                        momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.STEPSIZE, gamma=cfg.TRAIN.GAMMA)
        # load the relative hyperpremeter
        criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, use_gpu)


        # get dataset size
        epoch_size = len(self.dataset) // cfg.TRAIN.BATCH_SIZE
        data_loader = data.DataLoader(self.dataset, cfg.TRAIN.BATCH_SIZE, num_workers=4,
                                  shuffle=True, collate_fn=dataset_factory.detection_collate) #, pin_memory=True)
        for epoch in iter(range(start_epoch+1, self.max_epochs)):
            #learning rate
            exp_lr_scheduler.step(epoch)
            #batch data
            batch_iterator = iter(data_loader)
            loc_loss = 0
            conf_loss = 0
            for iteration in iter(range((epoch_size))):
                images, targets = next(batch_iterator)
                print(targets)
                if use_gpu:
                    images = Variable(images.cuda())
                    targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(anno, volatile=True) for anno in targets]

                # forward
                out = self.net(images, is_train=True)

                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets, self.priors)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()

                loc_loss += loss_l.data[0]
                conf_loss += loss_c.data[0]

                self.add_summary(epoch, iteration, epoch_size, loss_l.data[0], loss_c.data[0], optimizer)

                if iteration % 100 == 99:
                    return True
            if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                self.save_checkpoints(epoch)

        
    def add_summary(self, epoch, iters, epoch_size, loc_loss, conf_loss, optim):
        if iters == 0:
            sys.stdout.write('\n')
        prograss = iters/epoch_size
        lr = optim.param_groups[0]['lr']
        log = '\rEpoch {epoch:3d}: {percent:.2f}%[{prograss}] || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\r'.format(epoch=epoch, lr=lr,
                prograss='#'*int(round(10*prograss)) + '-'*int(round(10*(1-prograss))), percent=prograss, loc_loss=loc_loss, conf_loss=conf_loss)
        sys.stdout.write(log)
        sys.stdout.flush()
        return True

    def predict(self, img):
        return True



def train_model(base_fn=None, model_fn=None, max_epochs=None, output_dir=None, tbdir=None, resume_checkpoint=None, gpus=None, prior_box=None):
    sw = SolverWrapper(base_fn, model_fn, max_epochs, output_dir, tbdir, resume_checkpoint, gpus, prior_box)
    sw.train_model()
    return True