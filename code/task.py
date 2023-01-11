import models
import torch
import torch.nn as nn


class Task:
    def __init__(self, name, is_frozen, model, optimizer, scheduler) -> None:
        super().__init__()
        self.model = model
        self.is_frozen = is_frozen
        self.name = name
        self.optimizer = optimizer
        self.scheduler = scheduler

    def zero_grad(self):
        if self.is_frozen:
            pass
        else:
            self.optimizer.zero_grad()

    def step(self):
        if self.is_frozen:
            pass
        else:
            self.optimizer.step()

    def forward(self, **args):
        if self.is_frozen:
            with torch.no_grad():
                output = self.model(**args)
        else:
            output = self.model(**args)
        return output

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


def init_task(config, name, is_frozen, step, type, lr, device):

    model = getattr(models, type)(config)
    model = model.to(device)
    learning_rate = lr
    params = list(model.parameters())
    print(name)
    # SGD optimizer for all networks
    if is_frozen:
        optimizer = None
        scheduler = None
    else:

        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.sch_milestones,
                                                     gamma=config.sch_gamma) if step else None
        print(optimizer.param_groups[0]['lr'])
    task = Task(name, is_frozen, model, optimizer, scheduler)
    return task
