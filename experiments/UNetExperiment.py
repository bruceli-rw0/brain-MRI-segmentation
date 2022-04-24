"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from medpy.io import save

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard
from utils.volume_stats import Dice3d, Jaccard3d
from networks.RecursiveUNet import UNet
from inference.UNetInferenceAgent import UNetInferenceAgent

class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    The basic life cycle of a UNetExperiment is:

        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, opt, logger, split, dataset):
        self.logger = logger
        self.n_epochs = opt.num_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = opt.model_name

        # Create output folders
        Path(opt.root_dir/opt.ckp_dir).mkdir(exist_ok=True)
        self.out_dir = opt.root_dir/opt.ckp_dir/f'{opt.timeID}-{self.name}'
        self.out_dir.mkdir(exist_ok=True)

        # Create data loaders
        self.train_loader = DataLoader(
            SlicesDataset(dataset[split["train"]]),
            batch_size=opt.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.val_loader = DataLoader(
            SlicesDataset(dataset[split["val"]]),
            batch_size=1, 
            num_workers=0
        )
        self.test_data = dataset[split["test"]]

        if not torch.cuda.is_available():
            logger.warning("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The recursive UNet model is from German Cancer Research Center, 
        # Division of Medical Image Computing. It is quite complicated and works 
        # very well on this task.
        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        # use a standard cross-entropy loss since the model output is essentially
        # a tensor with softmax'd prediction of each pixel's probability of belonging 
        # to a certain class
        self.loss_function = torch.nn.CrossEntropyLoss()

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        # Scheduler helps us update learning rate automatically
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # Set up Tensorboard. By default it saves data into 'runs' folder.
        self.tensorboard_train_writer = SummaryWriter(log_dir=opt.root_dir/opt.log_dir/opt.timeID, comment="_train")
        self.tensorboard_val_writer = SummaryWriter(log_dir=opt.root_dir/opt.log_dir/opt.timeID, comment="_val")

    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        self.logger.info(f"Training epoch {self.epoch}...")
        self.model.train()

        # Loop over our minibatches
        for i, batch in enumerate(tqdm(self.train_loader, desc="Trainloader", position=0, leave=True)):
            self.optimizer.zero_grad()

            # Feed data to the model and feed target to the loss function
            data = batch['image'].to(self.device)
            target = batch['seg'].to(self.device)

            prediction = self.model(data)

            # get softmax'd version of prediction to output a probability map
            # to visualize how the model converges to the solution
            prediction_softmax = F.softmax(prediction, dim=1)
            loss = self.loss_function(prediction, target[:, 0, :, :])

            # What does each dimension of variable prediction represent?
            """
            1st dim == batch size
            2nd dim == output for each class
            3rd dim == patch size
            4th dim == patch size
            """

            loss.backward()
            self.optimizer.step()

            if (i % 100) == 0:
                # Output to console on every 30th batch
                self.logger.info(f"Epoch: {self.epoch} Train loss: {loss}")
                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                # You don't need to do anything with this function, but you are welcome to 
                # check it out if you want to see how images are logged to Tensorboard
                # or if you want to output additional debug data
                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter
                )
        self.logger.info(f"Epoch {self.epoch} training complete")

    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """
        self.logger.info(f"Validating epoch {self.epoch}...")

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="Validloader", position=0, leave=True)):
                
                data = batch['image'].to(self.device)
                target = batch['seg'].to(self.device)

                prediction = self.model(data)
                prediction_softmax = F.softmax(prediction, dim=1)
                loss = self.loss_function(prediction, target[:, 0, :, :])

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax, 
            prediction,
            (self.epoch+1) * 100
        )
        self.logger.info(f"Epoch {self.epoch} validation complete")

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")
        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        self.logger.info("Testing...")
        self.model.eval()

        # In this method we will be computing metrics that are relevant to the task of 3D volume
        # segmentation. Therefore, unlike train and validation methods, we will do inferences
        # on full 3D volumes, much like we will be doing it when we deploy the model in the 
        # clinical environment. 
        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []

        # for every in test set
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            # average Jaccard on test set should be around 0.80
            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            # STAND-OUT SUGGESTION: By way of exercise, consider also outputting:
            # * Sensitivity and specificity (and explain semantic meaning in terms of 
            #   under/over segmenting)
            # * Dice-per-slice and render combined slices with lowest and highest DpS
            # * Dice per class (anterior/posterior)

            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc
            })
            self.logger.info(f"{x['filename']} Dice {dc:.4f}; Jaccard {jc:4f}")
            save(pred_label, f"_prediction/{x['filename']}")

        out_dict["overall"] = {
            "mean_dice": np.mean(dc_list),
            "mean_jaccard": np.mean(jc_list)
        }

        self.logger.info("Testing complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        self.logger.info("Experiment started.")

        # Iterate over epochs
        for self.epoch in tqdm(range(1,self.n_epochs+1), desc="Epoch", position=0, leave=True):
            self.train()
            self.validate()

        # save model for inferencing
        self.save_model_parameters()

        self._time_end = time.time()
        self.logger.info(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.localtime(self._time_end - self._time_start))}")
