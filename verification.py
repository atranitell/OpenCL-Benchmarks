

import torch
from torch import nn

def run():
  layer = nn.Conv2d(512, 1024, 3, 1, 1, 1);
  layer.weight.data.fill_(1.5)
  layer.bias.data.fill_(0.5)

  inputs = torch.zeros([1, 512, 13, 13])
  for i in range(512):
    inputs[0][i].fill_(0.01*i)

  print(inputs.sum())

  layer.eval()
  with torch.no_grad():
    outputs = layer.forward(inputs)
    print(outputs.sum(), outputs.mean())
    for i in range(13):
      for j in range(13):
        print(i, j, outputs[0][0][i][j])

if __name__ == "__main__":
  run()
