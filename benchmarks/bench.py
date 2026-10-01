import time

import torch
from torch import nn

from nca import NCA

x = torch.rand(8, 4, 256, 256)
for inv in (False, True):
    nca = NCA("moore_r2", invariant=inv)
    net = nn.Sequential(nn.Linear(nca.in_features, 64), nn.ReLU(), nn.Linear(64, 4))
    with torch.no_grad():
        nca.process(net, x)
        t = time.perf_counter()
        for _ in range(5):
            nca.process(net, x)
        pt = (time.perf_counter() - t) / 5
        t = time.perf_counter()
        for _ in range(5):
            nca.perceive(x)
        pp = (time.perf_counter() - t) / 5
    print(f"invariant={inv} in_features={nca.in_features} process={pt*1e3:.0f}ms perceive={pp*1e3:.0f}ms")
