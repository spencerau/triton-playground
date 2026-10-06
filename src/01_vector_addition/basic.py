import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# print(torch.cuda.get_device_name(DEVICE))

torch.cuda.set_device(0)  # must come before get_active_torch_device()
DEVICE = triton.runtime.driver.active.get_active_torch_device()
print(DEVICE)

props = torch.cuda.get_device_properties(DEVICE)
print(props.name)               # gpu name (R9700 AI PRO etc)
print(props.gcnArchName)        # architecture name (gfx)
print(props.total_memory / 2**30, "GiB")

# print(props)

