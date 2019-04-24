import sys
import os
import argparse

sys.path.append(os.getcwd())

from ssds.core.config import cfg_from_file
from ssds.solver import Solver

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('-cfg', '--config', dest='config_file',
            help='optional config file', default=None, type=str, )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    
    cfg = cfg_from_file(args.config_file)
    train_solver = Solver(cfg)

    # train_model()

if __name__ == '__main__':
    train()
