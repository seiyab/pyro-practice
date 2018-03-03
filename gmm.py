from functools import reduce

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MyChain(nn.Module):
    def __init__(self):
        super(MyChain, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        pipe = [
            self.fc1,
            F.relu,
            self.fc2
        ]
        return reduce(lambda u, f: f(u), pipe, x)

my_chain = MyChain()

def main():
    ## ---- generate data ----
    mu1 = torch.Tensor([1, 1])
    mu2 = torch.Tensor([-1, -1])

    latent_clusters = torch.Tensor([[0] for _ in range(32)] + [[1] for _ in range(32)])
    data = Variable(torch.randn([64, 2]) * 0.5 + latent_clusters * mu1 + (1 - latent_clusters) * mu2)

    ## ---- setup inference ----
    optimizer = pyro.optim.Adam({'lr': 0.01})

    svi = pyro.infer.SVI(model, guide, optimizer, loss='ELBO')

    ## ---- inference ----
    n_steps = 3000
    for step in range(n_steps):
        svi.step(data)
        if step % (n_steps // 10) == 0:
            print('.', end='', flush=True)


    ## ---- plot ----
    data_np = data.data.numpy()

    x_min, x_max, y_min, y_max = data_np[:,0].min(), data_np[:,0].max(), data_np[:,1].min(), data_np[:,1].max()
    xx, yy = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1), np.linspace(y_min-0.1, y_max+0.1))

    grids = Variable(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]))
    Z = (my_chain(grids).data.numpy() < 0).reshape(xx.shape) * 1
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    data_1 = data[:32].data.numpy()
    data_2 = data[32:].data.numpy()
    plt.plot(data_1[:,0], data_1[:,1], '.', color='black')
    plt.plot(data_2[:,0], data_2[:,1], '.', color='white')
    plt.savefig('foo.png')

def model(data):
    mu_0 = Variable(torch.Tensor([0, 0]), requires_grad=False)
    sigma_0 = Variable(torch.Tensor([1, 1]), requires_grad=False)

    mu_1 = pyro.sample('mu_z1', dist.normal, mu_0, sigma_0)
    mu_2 = pyro.sample('mu_z2', dist.normal, mu_0, sigma_0)
    sigma = Variable(torch.Tensor([0.5, 0.5]), requires_grad=False)

    for i in pyro.irange('data_loop', len(data)):
        cluster = pyro.sample('cluster_{}'.format(i), dist.bernoulli, Variable(torch.Tensor([0.5])))
        pyro.observe('obs_{}'.format(i), dist.normal, data[i], mu_1 if cluster.data[0] < 0.5 else mu_2, sigma)

def guide(data):
    pyro.module('my_chain', my_chain)

    sigma = Variable(torch.Tensor([0.5, 0.5]), requires_grad=False)

    mu_param_1 = pyro.param("mu_param_1", Variable(torch.randn([2]) * 0.1, requires_grad=True))
    mu_param_2 = pyro.param("mu_param_2", Variable(torch.randn([2]) * 0.1, requires_grad=True))

    mu_1 = pyro.sample('mu_z1', dist.normal, mu_param_1, sigma)
    mu_2 = pyro.sample('mu_z2', dist.normal, mu_param_2, sigma)

    zs = F.sigmoid(my_chain(data))

    for i in pyro.irange('data_loop', len(zs)):
        pyro.sample('cluster_{}'.format(i), dist.bernoulli, zs[i])

main()
