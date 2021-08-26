# Export a simple TorchScript model

import torch

class DemoModule(torch.nn.Module):
    def forward(self, data, incr):
        # type: (Tensor, float) -> Tensor
        # Add the incr
        larger = data + incr
        # Then reduce sum over dim 0
        thinner = torch.sum(larger, 0)
        return thinner

model = DemoModule()
model.eval()
smod = torch.jit.script(model)
smod.eval()
smod.save("demo-model.pt1")
loaded = torch.jit.load("demo-model.pt1")
loaded.eval()
loaded(torch.tensor([[1,2,3],[4,5,6]]), 3.0)
