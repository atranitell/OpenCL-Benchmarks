

import torch
from torch import nn

def run():
  layer = nn.Conv2d(512, 1024, 3, 1, 1, 1);
  layer.weight.data.fill_(1.5)
  layer.bias.data.fill_(0.5)

  inputs = torch.zeros([1, 512, 13, 13])
  inputs.fill_(0.25)

  layer.eval()
  with torch.no_grad():
    outputs = layer.forward(inputs)
    print(outputs.sum(), outputs.mean())

if __name__ == "__main__":
  run()
