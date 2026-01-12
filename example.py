import numpy as np
import pandas as pd
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import time
import os
from scipy.stats.stats import pearsonr
from helpers import *
from avi import *
from onestep import *
import torch.nn as nn


# simulate paramters and data (with function from helpers.py)
a, b, theta, Q = simpars()
data = simdata(a, b, theta)

# remove previous logging file if necessary (logs loss)
if os.path.exists('logs/example/version_0/metrics.csv'):
    os.remove('logs/example/version_0/metrics.csv')

# create logger and trainer objects
logger = CSVLogger("logs", name='example', version=0)
trainer = Trainer(max_epochs=4000,
                  enable_checkpointing=False,
                  logger=logger,
                  callbacks=[
                      EarlyStopping(monitor='train_loss',
                                    min_delta=.00000008,
                                    patience=50,
                                    mode='min')])

# create dataset and data loader objects
dataset = SimDataset(data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# initialize model
vae = VAE(nitems=data.shape[1],
          dataloader=train_loader,
          latent_dims=Q.shape[1],
          hidden_layer_size=(data.shape[1] + Q.shape[1]) // 2,
          qm=Q,
          learning_rate=.01,
          n_samples=25)

# fit model and track time
start = time.time()
trainer.fit(vae)
runtime_vae = time.time() - start

# extract paramter estimates
a_est = vae.decoder.weights.t().detach().numpy()
b_est = vae.decoder.bias.t().detach().numpy()

# compute information matrix and perform one step
start = time.time()
onestep_a, onestep_b, avi_se_a, avi_se_b = onestep(vae, data, nquad=15, se_type='fisher', batch_size=1)
runtime_onestep = time.time() - start
print(f'one-step time: {runtime_onestep}')

# compute information again at new estimates (in order to get standard errors)
vae.decoder.weights = nn.Parameter(onestep_a.T)
vae.decoder.bias = nn.Parameter(onestep_b)
start = time.time()
_, _, onestep_se_a, onestep_se_b = onestep(vae, data, nquad=15, se_type='fisher', batch_size=1)
runtime_standard_errors = time.time() - start
print(f'Time computing standard errorr (FIM + sandwich estimator): {runtime_standard_errors}')

# invert slopes if necessary
for dim in range(a_est.shape[1]):
    if pearsonr(a_est[:, dim], a[:, dim])[0] < 0:
        a_est[:, dim] *= -1
        onestep_a[:, dim] *= -1


print(f'MSE(true slopes, AVI slopes): {MSE(a[a!=0], a_est[a!=0])}')
print(f'MSE(true slopes, One-step slopes): {MSE(a[a!=0], onestep_a.detach().numpy()[a!=0])}')

print(f'MSE(true intercepts, AVI intercepts): {MSE(b, b_est)}')
print(f'MSE(true intercepts, One-step intercepts): {MSE(b, onestep_b.detach().numpy())}')