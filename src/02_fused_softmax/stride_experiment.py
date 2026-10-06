import torch

import triton
import triton.language as tl
from triton.runtime import driver

# torch.cuda.set_device(1)
DEVICE = triton.runtime.driver.active.get_active_torch_device()

# cant really set gpu via code for some reason so use env var
# HIP_VISIBLE_DEVICES=1 python src/02_fused_softmax/fused_softmax.py

props = torch.cuda.get_device_properties(DEVICE)
print(props.name)


@triton.jit # triton decorator means "compile function into Triton GPU Code"
# function/code that actually computes softmax
def softmax_kernel(output_ptr, input_ptr, 
                   input_row_stride, output_row_stride, 
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        #print("pid", row_start, "row", row_idx)

        # stride presents how much we increase ptr to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # block size is the next power of 2 greater than n_cols, 
        # so we can fit each row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        # load row into SRAM with mask since BLOCK_SIZE may be > than n_cols
        # mask marks which slots are "real", and then loads the "fake" ones with -inf in other=
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # subtract max for numerical stability
        row_minus_max = row - tl.max(row, axis=0)

        # NOTE: exponentiation in Triton is fast but aproximate (like think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # write output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
# fixed function to account for differing col stride
def softmax_kernel_fixed(output_ptr, input_ptr, 
                        input_row_stride, output_row_stride,
                        input_col_stride, output_col_stride,
                        n_rows, n_cols,
                        BLOCK_SIZE: tl.constexpr,
                        num_stages: tl.constexpr):
    # starting row
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        #print("pid", row_start, "row", row_idx)

        # stride presents how much we increase ptr to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # block size is the next power of 2 greater than n_cols, 
        # so we can fit each row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets * input_col_stride

        # load row into SRAM with mask since BLOCK_SIZE may be > than n_cols
        # mask marks which slots are "real", and then loads the "fake" ones with -inf in other=
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # subtract max for numerical stability
        row_minus_max = row - tl.max(row, axis=0)

        # NOTE: exponentiation in Triton is fast but aproximate (like think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # write output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets * output_col_stride
        tl.store(output_ptrs, softmax_output, mask=mask)


print("\n--- stride experiment ---")
# 1D tensor [0, 1, 2, ... 39] and reshapes into 4 rows of 10 cols
base = torch.arange(40, dtype=torch.float32, device=DEVICE).reshape(4, 10)

print(base)

# take every row and every other col so -> shape (4, 5) of evens
# memory is still the same so this causes a bug
x = base[:, ::2]
print("x shape:", x.shape, "stride:", x.stride())
print(x)
y = torch.empty(x.shape, device=DEVICE)   # normal contiguous output
z = torch.empty(x.shape, device=DEVICE)

# computes softmax of [0, 1, 2, 3, 4]s -> incorrect due to stale mem addresses and stride is only 1
softmax_kernel[(1,)](y, x, x.stride(0), y.stride(0), 4, 5, BLOCK_SIZE=8, num_stages=1)

print("triton:", y)
# computes softmax of [0, 2, 4, 6, 8]
print("torch: ", torch.softmax(x, dim=1))

softmax_kernel_fixed[(1,)](z, x, 
                           x.stride(0), z.stride(0), 
                           x.stride(1), z.stride(1),
                           4, 5, 
                           BLOCK_SIZE=8, 
                           num_stages=1)
print("triton with correct col stride:", z)
