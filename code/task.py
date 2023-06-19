import models
import torch
import torch.nn as nn


class Task:
    """
        A class representing a task for training and evaluation.

        Attributes:
            model (torch.nn.Module): The model for the task.
            name (str): The name of the task.
            optimizer (torch.optim.Optimizer): The optimizer for the task.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler for the task.
        """

    def __init__(self, name, model, optimizer, scheduler) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.optimizer = optimizer
        self.scheduler = scheduler

    def zero_grad(self):
        if self.optimizer != None:
            self.optimizer.zero_grad()

    def step(self):
        if self.optimizer != None:
            self.optimizer.step()

    def forward(self, **args):
        output = self.model(**args)
        return output

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()


def init_task(config, name, step, type, lr, device):
    model = getattr(models, type)(config)
    model = model.to(device)
    learning_rate = lr
    params = list(model.parameters())
    try:
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.sch_milestones,
                                                         gamma=config.sch_gamma) if step else None
    except:
        optimizer = None
        scheduler = None

    task = Task(name, model, optimizer, scheduler)
    return task
