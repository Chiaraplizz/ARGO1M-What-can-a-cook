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
# import clip
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
        print(self.name)
        self.wb = wandb.init(config=self.args.config, mode=self.args.wandb_mode, name=self.name, project=date)
        self.config = self.wb.config
        self.use_text = self.config.use_text
        self.last_epoch = self.config.last_epoch if self.config.resume else 0
        print('use_text: ', self.use_text)
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
            # Bug with ffcv ToDevice so do it within the training loop instead.
            # pipelines={
            #       'feat': [NDArrayDecoder(), ToTensor(), ToDevice(self.device)],
            #       'label': [NDArrayDecoder(), ToTensor(), ToDevice(self.device)]
            #     }

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

        # use for dataset stats
        self.train_ds = classification_dataset(self.config, split="train")
        self.running_stats = None

    def init_model(self):

        # initiate
        if (self.config.n_gpu > 0) and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.models = [init_task(self.config, n, f, s, t, lr, self.device) for t, n, f, s, lr in
                       zip(self.config.model_types, self.config.model_names, self.config.freeze_models,
                           self.config.step,
                           self.config.model_lrs)]

        self.losses = [init_loss(l, self.config, self.train_ds) for l in self.config.loss_types]
        self.losses_weights = self.config.loss_weights

        for model in self.models:
            self.wb.watch(model.model)

    def training_run(self):
        acc_min = 0
        update = False
        final_acc = {}

        if self.config.resume:

            for i in range(len(self.models)):
                weights_dir = '/user/work/qh22492/saved_models/25-10-2022/'+self.name
                try:
                    checkpoint = torch.load(
                        os.path.join(weights_dir,
                                     self.name + '_' + str(self.config.last_epoch) + '_' + self.config.model_names[i] + '.pth'))
                    # import pdb; pdb.set_trace()
                    self.models[i].model.load_state_dict(checkpoint[self.config.model_names[i]])

                    self.models[i].optimizer.load_state_dict(checkpoint[self.config.model_names[i] + '_opt'])
                except:
                    print('Not found: ' + os.path.join(weights_dir,
                                                       self.name + '_' + str(self.config.last_epoch) + '_' + self.config.model_names[
                                                           i] + '.pth'))
            print('MODEL RESUME ')
        for epoch in range(self.last_epoch, self.config.epochs):

            if (self.config.update_stat_fraction is not None) and (epoch >= self.config.stat_update_start_epoch):
                self.update_stats()
            print('updated')
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

                #
                print(accs)

            self.wb.log(log_dict)

    def train_epoch(self, epoch):
        # check that the dir does not exists
        start = timeit.default_timer()
        train_losses = {k: np.float64(0.0) for k in self.config.loss_names}
        train_losses["loss"] = np.float64(0.0)
        start = 0
        if self.config.curriculum_learning:
            for dict in self.config.stage:
                for key in dict.keys():
                    if epoch >= start and epoch < dict[key]:
                        attn = key
                        start = dict[key]
                    else:
                        continue
        else:
            attn = self.config.gen_attn

        for batch_idx, input in tqdm(enumerate(self.loaders["train"])):
            break
            if self.config.save_matrices:
                save_flag = True
            else:
                save_flag = False

            output_collection = create_input(self.config, self.use_text, input, self.device)

            for model in self.models:
                model.zero_grad()

            # loop through all models in order
            for i in range(len(self.config.model_names)):
                if self.config.model_use_train[i]:
                    # collect inputs to this model as a dictionary
                    # e.g. "x": tensor1, "target": tensor2
                    inputs = {k: output_collection[v] for k, v in self.config.model_inputs[i].items()}

                    if self.config.model_names[i] == 'Classifier_mix':
                        inputs['uid'] = output_collection['data.uid']

                    if 'attn' in inputs.keys():
                        inputs['attn'] = attn
                        inputs['uid'] = output_collection['data.uid']
                        inputs['sim_1_arg'] = self.config.sim_1
                        inputs['sim_2_arg'] = self.config.sim_2
                        if self.config.alternate:
                            inputs[
                                'attn'] = 'same scenario different source' if batch_idx % 2 == 0 else 'same source different scenario'
                        if 'CLIP_scenario' == self.config.model_names[i]:
                            inputs['attn'] = 'other sources'
                            inputs['sim_1_arg'] = self.config.sim_1_1
                            inputs['sim_2_arg'] = self.config.sim_2_1
                        if 'CLIP_source' == self.config.model_names[i]:
                            inputs['attn'] = 'other scenarios'
                            inputs['sim_1_arg'] = self.config.sim_1_2
                            inputs['sim_2_arg'] = self.config.sim_2_2


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
            if save_flag:
                self.save_matrices(input, output_collection['CLIP.weights'], batch_idx, epoch, 'train')
            model_prediction = output_collection[self.config.model_prediction]

            total_loss = 0
            for i in range(len(self.config.loss_names)):
                inputs = {k: output_collection[v] for k, v in self.config.loss_inputs[i].items()}
                batch_loss = self.losses_weights[i] * self.losses[i](**inputs)
                name = self.config.loss_names[i]
                train_losses[name] = train_losses[name] + batch_loss.item()
                total_loss += batch_loss
                train_losses["loss"] = train_losses["loss"] + total_loss.item()
            # total_loss /= len(self.config.loss_names)
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
        if self.config.save_model and ((epoch == 49) or (epoch % 10)==0):
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
                output_collection = create_input(self.config, True, input, self.device)


                if self.config.save_matrices and (epoch % 10 == 0 or epoch == 49):
                    save_flag = True
                else:
                    save_flag = False

                # loop through all models in order
                for i in range(len(self.config.model_names)):
                    if self.config.model_use_eval[i]:
                        # collect inputs to this model as a dictionary
                        # e.g. "x": tensor1, "target": tensor2
                        inputs = {k: output_collection[v] for k, v in self.config.model_inputs[i].items()}
                        # expands dictionary so model is passed x=tensor1, target=tensor2
                        if self.config.model_names[i] == 'Classifier_mix' or self.config.model_names[i] == 'CLIP':
                            inputs['uid'] = output_collection['data.uid']

                        outputs = self.models[i].forward(**inputs)

                        for k, v in outputs.items():
                            # prepends model name that has provided the output.
                            # e.g. "logits" -> "mlp.logits"
                            output_collection["{}.{}".format(self.config.model_names[i], k)] = v
                if save_flag:
                    self.save_matrices(output_collection['data.target'], output_collection['CLIP.weights'], batch_idx, epoch, split)
                if self.config.resume and False:
                    path = '/user/work/qh22492/extracted_feat/' + self.name + '/' + split + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    for i in range(128):
                        torch.save(output_collection['MLP.representations'][i],
                                   path +
                                   output_collection['data.uid'][i][
                                       0] + '_' + str(output_collection['data.target'][i, 0].item())
                                   + '_' + str(output_collection['data.target'][i, 1].item()) + '_' +
                                   str(output_collection['data.target'][i, 2].item()) + '_' + str(batch_idx)+'_'+str(i) + '.pt')

                model_prediction = output_collection[self.config.model_prediction]
                all_labels.append(output_collection['data.target'].detach().clone())
                all_logits.append(model_prediction.detach())
                #print(all_logits)
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)


            acc = accuracy_measures(self.config, all_logits, all_labels, split)
            acc_dict = {"{} {}".format(split, k): acc[k] for k in acc.keys()}

        for i in range(len(self.models)):
            self.models[i].train()
        return acc_dict

    def save_matrices(self, input, matrix, batch_idx, epoch, split):
        if batch_idx == 0:
            self.target = input
            self.matrix  = matrix.unsqueeze(0)
        else:
            self.target=torch.cat((self.target, input), dim=0)
            self.matrix=torch.cat((self.matrix, matrix.unsqueeze(0)), dim=0)
        path = self.config.store_dir + self.name + '/' + 'tot/' + '_' + split
        if not os.path.exists(path):
            os.makedirs(path)
        os.chdir(path)
        torch.save(self.target, "input_tot_" + split + ".pt")
        #torch.save(input, "input" + "_" + str(batch_idx) + ".pt")
        #torch.save(matrix, "matrix" + "_" + str(batch_idx) + ".pt")
        torch.save(self.matrix, "matrix_tot_"+split+".pt")

    def update_stats(self, split="train"):
        for i in range(len(self.models)):
            self.models[i].eval()

        with torch.no_grad():
            all_reps = []
            all_labels = []
            for batch_idx, input in enumerate(self.loaders[split]):
                output_collection = create_input(self.config, self.use_text, input, self.device)
                #if (batch_idx / len(self.loaders[split])) > self.config.update_stat_fraction:
                #    break
                if torch.rand(1) > self.config.update_stat_fraction:
                    continue

                # loop through all models in order
                inputs = {k: output_collection[v] for k, v in self.config.model_inputs[0].items()}
                # expands dictionary so model is passed x=tensor1, target=tensor2
                outputs = self.models[0].forward(**inputs)

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
                self.running_stats = {k: self.running_stats[k] * self.config.update_stat_alpha + epoch_stats[k] * (1 - self.config.update_stat_alpha) for k in epoch_stats}

        for i in range(len(self.models)):
            self.models[i].train()

def main():
    learner = Learner()
    learner.training_run()


if __name__ == "__main__":
    main()
