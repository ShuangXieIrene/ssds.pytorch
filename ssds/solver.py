import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import sys
from tensorboardX import SummaryWriter

from ssds.core import optimizer
from ssds.core import checkpoint
from ssds.core.criterion import configure_criterion
from ssds.modeling import model_builder
from ssds import pipeline
from ssds.dataset.dataset_factory import load_data

class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load data
        print('===> Loading data')
        self.train_loader = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
        # self.eval_loader = load_data(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        # self.test_loader = load_data(cfg.DATASET, 'test') if 'test' in cfg.PHASE else None
        # self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        # Build model
        print('===> Building model')
        self.model = model_builder.create_model(cfg.MODEL)

        # Utilize GPUs for computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print the model architecture and parameters
        print('Model architectures:\n{}\n'.format(self.model))

        # print('Parameters and size:')
        # for name, param in self.model.named_parameters():
        #     print('{}: {}'.format(name, list(param.size())))

        # print trainable scope
        print('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        trainable_param_ = optimizer.trainable_param(self.model, cfg.TRAIN.TRAINABLE_SCOPE)
        self.optimizer = optimizer.configure_optimizer(trainable_param_, cfg.TRAIN.OPTIMIZER)
        self.exp_lr_scheduler = optimizer.configure_lr_scheduler(self.optimizer, cfg.TRAIN.LR_SCHEDULER)
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
        self.criterion = configure_criterion(cfg.TRAIN.CRITERION)(self.cfg.MATCHER, self.device)

        # Set the logger
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    def train_model(self):
        previous = checkpoint.find_previous_checkpoint(self.cfg.EXP_DIR)
        if previous:
            start_epoch = previous[0][-1]
            checkpoint.resume_checkpoint(self.model, previous[1][-1], self.cfg.TRAIN.RESUME_SCOPE)
        else:
            start_epoch = 0
            if self.cfg.RESUME_CHECKPOINT:
                print('Loading initial model weights from {:s}'.format(self.cfg.RESUME_CHECKPOINT))
                checkpoint.resume_checkpoint(self.model, self.cfg.RESUME_CHECKPOINT, self.cfg.TRAIN.RESUME_SCOPE)

        if torch.cuda.is_available():
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            if self.cfg.DEVICE_ID:
                self.model = nn.DataParallel(self.model, device_ids=self.cfg.DEVICE_ID)
            cudnn.benchmark = True
        self.model.to(self.device)

        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
            #learning rate
            sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
            if epoch > warm_up:
                self.exp_lr_scheduler.step(epoch-warm_up)

            # start phases for epoch
            if 'train_anchor_basic' in self.cfg.PHASE:
                # TODO: Change cfg.MODEL to cfg.PRIOR
                priorbox = model_builder.create_priors(self.cfg.MODEL, self.model, self.cfg.IMAGE_SIZE)
                priors = priorbox().to(self.device)

                pipeline.train_anchor_basic(self.model, self.train_loader, self.optimizer, self.criterion, priors, self.writer, epoch, self.device)

            # save checkpoint
            if epoch % self.cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                checkpoint.save_checkpoints(self.model, self.cfg.EXP_DIR, self.cfg.CHECKPOINTS_PREFIX, epoch)



    # def export_graph(self):
    #     self.model.train(False)
    #     # dummy_input = Variable(torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).cuda()
    #     # Export the model
    #     torch_out = torch.onnx._export(self.model,             # model being run
    #                                    dummy_input,            # model input (or a tuple for multiple inputs)
    #                                    "graph.onnx",           # where to save the model (can be a file or file-like object)
    #                                    export_params=True)     # store the trained parameter weights inside the model file
        # if not os.path.exists(cfg.EXP_DIR):
        #     os.makedirs(cfg.EXP_DIR)
        # self.writer.add_graph(self.model, (dummy_input, ))
