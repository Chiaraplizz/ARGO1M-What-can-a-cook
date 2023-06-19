import torch
import ffcv
from ffcv.loader import Loader, OrderOption
from ffcv.fields import JSONField
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor, ToDevice
from ffcv.fields.bytes import BytesDecoder
import pdb
from tqdm import tqdm
import argparse
import wandb
import timeit
import os
import numpy as np
import time
import gc
from feature_dataset import classification_dataset
import models
from utils import accuracy_measures, get_seq_len_from_config, get_wandb_run_name_from_config_path, calculate_dist_stats, \
    preprocess_text, create_input, init_loss
from task import init_task
import losses

from datetime import date

today = date.today()
print("Today's date:", today)
date = today.strftime("%d-%m-%Y")
import random

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", default="configs/full.yaml", help="wandb config file.")
parser.add_argument("--wandb_mode", "-wb", choices=["online", "offline", "disabled"], default="online",
                    help="disable wandb logging")
args = parser.parse_args()
print(args.config)
wb = wandb.init(config=args.config, mode=args.wandb_mode, name=get_wandb_run_name_from_config_path(args.config),
                project=date)
config = wb.config
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
CUDA_LAUNCH_BLOCKING = 1


class Learner:
    def __init__(self):
        self.parse_command_line()
        self.name = get_wandb_run_name_from_config_path(self.args.config)
        self.wb = wandb.init(config=self.args.config, mode=self.args.wandb_mode, name=self.name, project=date)
        self.config = self.wb.config
        self.use_text = self.config.use_text
        self.last_epoch = self.config.last_epoch if self.config.resume else 0
        self.init_data()
        self.init_model()

    def __del__(self):
        self.wb.finish()

    def parse_command_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "-c", default="configs/full.yaml", help="wandb config file.")
        parser.add_argument("--wandb_mode", "-wb", choices=["online", "offline", "disabled"], default="online",
                            help="disable wandb logging")
        self.args = parser.parse_args()

    def init_data(self):
        self.loaders = {}

        if self.config.use_ffcv:
            if self.use_text:
                pipelines = {
                    'feat': [NDArrayDecoder(), ToTensor()],
                    'narration': [BytesDecoder()],
                    'noun': [BytesDecoder()],
                    'label': [NDArrayDecoder(), ToTensor()]
                }
            else:
                pipelines = {
                    'feat': [NDArrayDecoder(), ToTensor()],
                    'label': [NDArrayDecoder(), ToTensor()]
                }

            for split_idx, split in enumerate(self.config.dataset_splits):
                ds_path = os.path.join(self.config.ffcv_path, self.config.dataset_ffcvs[split_idx])
                if split == "train":
                    order = getattr(OrderOption, self.config.ffcv_order)
                else:
                    order = OrderOption.SEQUENTIAL
                self.loaders[split] = Loader(ds_path, batch_size=self.config.batch_size,
                                             num_workers=self.config.n_workers, order=order, pipelines=pipelines,
                                             os_cache=self.config.ffcv_os_cache,
                                             batches_ahead=self.config.ffcv_batches_ahead, seed=self.config.seed)
        else:
            for split_idx, split in enumerate(self.config.dataset_splits):
                if split == "train":
                    shuffle = self.config.csv_shuffle
                else:
                    shuffle = False

                ds = classification_dataset(self.config, split=split)
                self.loaders[split] = torch.utils.data.DataLoader(ds, batch_size=self.config.batch_size,
                                                                  num_workers=self.config.n_workers, shuffle=shuffle)

        self.train_ds = classification_dataset(self.config, split="train")
        self.running_stats = None

    def init_model(self):
        if (self.config.n_gpu > 0) and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.models = [init_task(self.config, n, s, t, lr, self.device) for t, n, s, lr in
                       zip(self.config.model_types, self.config.model_names,
                           self.config.step,
                           self.config.model_lrs)]

        self.losses = [init_loss(l, self.config, self.train_ds) for l in self.config.loss_types]
        self.losses_weights = self.config.loss_weights

        for model in self.models:
            self.wb.watch(model.model)

    def training_run(self):
        acc_min = 0
        final_acc = {}

        if self.config.resume:
            for i in range(len(self.models)):
                weights_dir = self.config.weights_dir + self.name
                try:
                    checkpoint = torch.load(
                        os.path.join(weights_dir,
                                     self.name + '_' + str(self.config.last_epoch) + '_' + self.config.model_names[
                                         i] + '.pth'))
                    self.models[i].model.load_state_dict(checkpoint[self.config.model_names[i]], strict=True)
                    self.models[i].optimizer.load_state_dict(checkpoint[self.config.model_names[i] + '_opt'])
                except:
                    print('Not found: ' + os.path.join(weights_dir,
                                                       self.name + '_' + str(self.config.last_epoch) + '_' +
                                                       self.config.model_names[
                                                           i] + '.pth'))
        for epoch in range(self.last_epoch, self.config.epochs):
            update = False
            if (self.config.update_stat_fraction is not None) and (epoch >= self.config.stat_update_start_epoch):
                self.update_stats()
            log_dict = self.train_epoch(epoch)
            log_dict["epoch"] = epoch

            for s in self.config.dataset_splits:
                if s == "train":
                    continue
                accs = self.evaluate(s, epoch)

                for k in accs.keys():
                    log_dict[k] = accs[k]
                if s == "val" and accs['val accuracy'] > acc_min:
                    acc_min = accs['val accuracy']
                    update = True
                if update:
                    for k in accs.keys():
                        final_acc[k] = accs[k]

            self.wb.log(log_dict)

    def train_epoch(self, epoch):
        start = timeit.default_timer()
        train_losses = {k: np.float64(0.0) for k in self.config.loss_names}
        train_losses["loss"] = np.float64(0.0)
        start = 0

        for batch_idx, input in tqdm(enumerate(self.loaders["train"])):

            output_collection = create_input(self.use_text, input, self.device)

            for model in self.models:
                model.zero_grad()

            for i in range(len(self.config.model_names)):
                # loop through all models in order
                if self.config.model_use_train[i]:
                    # collect inputs to this model as a dictionary
                    # e.g. "x": tensor1, "target": tensor2
                    inputs = {k: output_collection[v] for k, v in self.config.model_inputs[i].items()}

                    # expands dictionary so model is passed x=tensor1, target=tensor2
                    outputs = self.models[i].forward(**inputs)

                    for k, v in outputs.items():
                        # prepends model name that has provided the output.
                        # e.g. "logits" -> "mlp.logits"
                        output_collection["{}.{}".format(self.config.model_names[i], k)] = v

            output_collection["epoch"] = epoch

            if (self.config.update_stat_fraction > 0) and (epoch >= self.config.stat_update_start_epoch):
                for k, v in self.running_stats.items():
                    output_collection[k] = v.to(self.device)
            else:
                output_collection['c_d_means'] = None

            total_loss = 0
            for i in range(len(self.config.loss_names)):
                inputs = {k: output_collection[v] for k, v in self.config.loss_inputs[i].items()}
                batch_loss = self.losses_weights[i] * self.losses[i](**inputs)
                name = self.config.loss_names[i]
                train_losses[name] = train_losses[name] + batch_loss.item()
                total_loss += batch_loss
                train_losses["loss"] = train_losses["loss"] + total_loss.item()
            total_loss.backward()

            for model in self.models:
                model.step()

        for model in self.models:
            if model.scheduler is not None:
                model.scheduler.step()

        stop = timeit.default_timer()
        epoch_time = stop - start
        print('Epoch time: ', epoch_time)
        print('Saving . . .')
        if self.config.save_model:
            weights_dir = os.path.join('./saved_models', date)
            weights_dir = os.path.join(weights_dir, self.name)

            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)
            try:
                for i in range(len(self.models)):
                    torch.save({self.config.model_names[i]: self.models[i].model.state_dict(),
                                self.config.model_names[i] + '_opt': self.models[i].optimizer.state_dict()
                                }, os.path.join(weights_dir,
                                                self.name + '_' + str(epoch) + '_' + self.config.model_names[
                                                    i] + '.pth'))

            except Exception as e:
                print("An error occurred while saving the checkpoint:")
                print(e)
        log_dict = {}
        for k in train_losses.keys():
            log_dict[k] = train_losses[k] / len(self.loaders["train"])
        log_dict["epoch time"] = epoch_time

        return log_dict

    def evaluate(self, split, epoch):

        for i in range(len(self.models)):
            self.models[i].eval()
        with torch.no_grad():
            all_labels = []
            all_logits = []

            for batch_idx, input in enumerate(self.loaders[split]):
                output_collection = create_input(False, input, self.device)

                # loop through all models in order
                for i in range(len(self.config.model_names)):
                    if self.config.model_use_eval[i]:
                        # collect inputs to this model as a dictionary
                        # e.g. "x": tensor1, "target": tensor2
                        inputs = {k: output_collection[v] for k, v in self.config.model_inputs[i].items() if
                                  not ('CIR' in v)}

                        # expands dictionary so model is passed x=tensor1, target=tensor2
                        outputs = self.models[i].forward(**inputs)

                        for k, v in outputs.items():
                            # prepends model name that has provided the output.
                            # e.g. "logits" -> "mlp.logits"
                            output_collection["{}.{}".format(self.config.model_names[i], k)] = v

                model_prediction = output_collection[self.config.model_prediction]
                all_labels.append(output_collection['data.target'].detach().clone())
                all_logits.append(model_prediction.detach())
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            acc = accuracy_measures(self.config, all_logits, all_labels)
            acc_dict = {"{} {}".format(split, k): acc[k] for k in acc.keys()}

        for i in range(len(self.models)):
            self.models[i].train()
        return acc_dict

    def update_stats(self, split="train"):
        for i in range(len(self.models)):
            self.models[i].eval()

        with torch.no_grad():
            all_reps = []
            all_labels = []
            for batch_idx, input in enumerate(self.loaders[split]):
                output_collection = create_input(False, input, self.device)

                if torch.rand(1) > self.config.update_stat_fraction:
                    continue

                # loop through MLP model 
                inputs = {k: output_collection[v] for k, v in
                          self.config.model_inputs[self.config.model_names.index('MLP')].items()}
                # expands dictionary so model is passed x=tensor1, target=tensor2
                outputs = self.models[self.config.model_names.index('MLP')].forward(**inputs)

                for k, v in outputs.items():
                    # prepends model name that has provided the output.
                    # e.g. "logits" -> "mlp.logits"
                    output_collection["{}.{}".format('MLP', k)] = v

                model_prediction = output_collection['MLP.representations']
                all_labels.append(output_collection['data.target'].detach())
                all_reps.append(model_prediction.detach())

            all_reps = torch.cat(all_reps, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            epoch_stats = calculate_dist_stats(self.config, all_reps, all_labels)

            if self.running_stats == None:
                self.running_stats = epoch_stats
            else:
                self.running_stats = {k: self.running_stats[k] * self.config.update_stat_alpha + epoch_stats[k] * (
                            1 - self.config.update_stat_alpha) for k in epoch_stats}

        for i in range(len(self.models)):
            self.models[i].train()


def main():
    learner = Learner()
    learner.training_run()


if __name__ == "__main__":
    main()
