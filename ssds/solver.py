import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

# from lib.layers import *
# from lib.utils.timer import Timer
# from lib.utils.data_augment import preproc
# from lib.modeling.model_builder import create_model
# from lib.dataset.dataset_factory import load_data
# from lib.utils.config_parse import cfg
# from lib.utils.eval_utils import *
# from lib.utils.visualize_utils import *

from ssds.core import optimizer
from ssds.core.criterion import configure_criterion
from ssds.modeling.model_builder import create_model
# from ssds.dataset.dataset_factory import load_data

class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load data
        print('===> Loading data')
        # self.train_loader = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
        # self.eval_loader = load_data(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        # self.test_loader = load_data(cfg.DATASET, 'test') if 'test' in cfg.PHASE else None
        # self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        # Build model
        print('===> Building model')
        self.model = create_model(cfg.MODEL)

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


    # def train_model(self):
    #     previous = self.find_previous()
    #     if previous:
    #         start_epoch = previous[0][-1]
    #         self.resume_checkpoint(previous[1][-1])
    #     else:
    #         start_epoch = self.initialize()

    #     # export graph for the model, onnx always not works
    #     # self.export_graph()

    #     # warm_up epoch
    #     warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
    #     for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
    #         #learning rate
    #         sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
    #         if epoch > warm_up:
    #             self.exp_lr_scheduler.step(epoch-warm_up)
    #         if 'train' in cfg.PHASE:
    #             self.train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu)
    #         if 'eval' in cfg.PHASE:
    #             self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
    #         if 'test' in cfg.PHASE:
    #             self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu)
    #         if 'visualize' in cfg.PHASE:
    #             self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)

    #         if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
    #             self.save_checkpoints(epoch)

    # def test_model(self):
    #     previous = self.find_previous()
    #     if previous:
    #         for epoch, resume_checkpoint in zip(previous[0], previous[1]):
    #             if self.cfg.TEST.TEST_SCOPE[0] <= epoch <= self.cfg.TEST.TEST_SCOPE[1]:
    #                 sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.cfg.TEST.TEST_SCOPE[1]))
    #                 self.resume_checkpoint(resume_checkpoint)
    #                 if 'eval' in cfg.PHASE:
    #                     self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
    #                 if 'test' in cfg.PHASE:
    #                     self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
    #                 if 'visualize' in cfg.PHASE:
    #                     self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)
    #     else:
    #         sys.stdout.write('\rCheckpoint {}:\n'.format(self.checkpoint))
    #         self.resume_checkpoint(self.checkpoint)
    #         if 'eval' in cfg.PHASE:
    #             self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, 0, self.use_gpu)
    #         if 'test' in cfg.PHASE:
    #             self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
    #         if 'visualize' in cfg.PHASE:
    #             self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, 0,  self.use_gpu)



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
