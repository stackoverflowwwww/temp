from email.policy import strict
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import numpy as np
import yaml
from u2pl.dataset import builder
import matplotlib.pyplot as plt
from u2pl.models.model_helper import ModelBuilder
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
writer=SummaryWriter("./runs/models")
cfg = yaml.load(open("config_test.yaml", "r"), Loader=yaml.Loader)
model = ModelBuilder(cfg["net"])
writer.add_graph(model,torch.zeros(1,3,513,513),use_strict_trace=False)
writer.close()