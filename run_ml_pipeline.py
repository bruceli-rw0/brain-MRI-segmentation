"""
This file contains code that will kick off training and testing processes
"""
import os
import time
import json
import logging
from pathlib import Path

import configargparse
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

logger = logging.getLogger(__name__)

def main(opt):
    # Load data
    logger.info("Loading data...")
    data = LoadHippocampusData(opt, y_shape=opt.patch_size, z_shape=opt.patch_size)

    # Create test-train-val split
    # In a real world scenario, should probably do multiple splits for 
    # multi-fold training to improve model quality
    keys = range(len(data))
    split = dict()
    
    keys_shuffle = np.random.permutation(keys)
    split['train'] = keys_shuffle[:int(len(keys)*0.7)]
    split['val'] = keys_shuffle[int(len(keys)*0.7):int(len(keys)*0.85)]
    split['test'] = keys_shuffle[int(len(keys)*0.85):]

    # Set up and run experiment
    exp = UNetExperiment(opt, logger, split, data)

    # free up memory
    del data
    # run training
    if opt.do_train:
        exp.run()

    if opt.pretrained_weight is not None:
        exp.load_model_parameters(f'{opt.pretrained_weight}/model.pth')
    # run testing
    if opt.do_test:
        results_json = exp.run_test()
    
    with open(exp.out_dir/"results.json", 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', type=str, is_config_file=True, help='Config file path.')

    parser.add_argument("--root_dir", type=str, default='.', help="Full path prefix")
    parser.add_argument("--data_dir", type=str, default='data', help="Directory to data")
    parser.add_argument("--log_dir", type=str, default='logging', help="Directory for saving training log")
    parser.add_argument("--ckp_dir", type=str, default='checkpoint', help="Directory for saving model checkpoint")

    parser.add_argument("--do_train", action='store_true', help="Train model")
    parser.add_argument("--do_test", action='store_true', help="Test model")

    parser.add_argument("--model_name", type=str, default='basic_unet', help="Model name")
    parser.add_argument("--pretrained_weight", type=str, default=None, help="Trained model weight")
    parser.add_argument("--epoch", type=int, default=0, help="The starting epoch")
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Size of the batches")
    parser.add_argument("--patch_size", type=int, default=64, help="Data input size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam: learning rate")
    
    opt = parser.parse_args()
    opt.timeID = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    opt.root_dir = Path(opt.root_dir)
    Path(opt.root_dir/opt.log_dir).mkdir(exist_ok=True)
    handlers = [
        logging.FileHandler(filename=Path(opt.root_dir)/opt.log_dir/f'{opt.timeID}-{opt.model_name}.txt', mode='w'),
        logging.StreamHandler()
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
    )
    main(opt)
