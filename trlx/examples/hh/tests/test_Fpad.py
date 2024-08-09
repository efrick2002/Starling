import torch.nn.functional as F
import torch

samples = [torch.zeros([2, 78]), torch.zeros([2, 100])]

len_max = max([samples[i].shape[1] for i in range(2)])
for i in range(2):
    samples[i] = F.pad(
        samples[i],
        (0, len_max - samples[i].shape[1]),
        value=32000,
    )

print(samples[0].shape)
print(samples[1].shape)
