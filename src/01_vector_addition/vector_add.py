import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr,       # ptr to first input vector
               y_ptr,       # ptr to second input vector
               output_ptr,  # ptr to output vector
               n_elements,  # size of vector (which vector?)
               BLOCK_SIZE: tl.constexpr,    # num of elements each program should process,
               # NOTE: 'constexpr' so it can be used as a shape value.
               ):
    # multiple programs in parallel processing different data. identify program using:
    pid = tl.program_id(axis=0) # 1d launch grid so axis=0

    # this program processes inputs that are offset from initial data
    # for example, if we have a vector of length 256 with block_size 64, the program accesses
    # elements in blocks of [0:64, 64:128, 128:192, 192:256]

    block_start = pid * BLOCK_SIZE

    # offsets is a list of pointers
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # create mask to guard memory operations against out of bound access
    mask = offsets < n_elements

    # print statement to visualize whats happening
    # print("pid", pid, "offsets", offsets, "mask", mask)

    # Load x and y from DRAM & mask out extra elements in case input is not a # multiple of block size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    # write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


# helper function that allocates z tensor and enques above kernel with appropriate grid/block sizes
def add(x: torch.Tensor, y: torch.Tensor):
    # preallocate output
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    # SPMD launch grid denotes the number of kernel instances that run in parallel
    # it's analogous to CUDA launch grids and can either be Tuple[int] or Callable[metaparameters] -> Tuple[int]
    # in this case, we use a 1D grid where size is the number of blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # NOTE:
    #   - each torch.tensor object is implicitly convert into a ptr to its first element
    #   - 'triton.jit'ed functions can be indexed with a launch rid to obtain a callable GPU kernel
    #   - dont forget to pass meta-parameters as keyword args
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    # we return a handle to z but since 'torch.cuda.synchronize()' hasn't been called, the kernel is still running
    # async at this point
    return output 


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)

print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')


#-- BENCHMARKING --
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))

def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, save_path="src/01_vector_addition/bench")


# print("\n--- Out of Bounds demo ---")
# n = 1
# buf = torch.full((2048,), -1.0, device=DEVICE)   # -1.0 marks memory we "don't own"
# a = torch.rand(n, device=DEVICE)
# b = torch.rand(n, device=DEVICE)
# out = buf[:n]                                     # output is a view into the start of buf

# add_kernel[(1,)](a, b, out, n, BLOCK_SIZE=1024)   # 1 program, covers indices 0..1023
# print(buf[995:1030])

# first 5 values are index 995, 996, 997, 998, 999 which are the last real sums a + b
# masks stops kernel from writing past n = 1000, so with mask, will be -1 at 1000+
# without mask, they end up being junk values


# print("\n--- grid demo ---")
# n = 10 # num elements
# block_size = 4
# a = torch.rand(n, device=DEVICE)
# b = torch.rand(n, device=DEVICE)
# out = torch.empty_like(a)
# add_kernel[(triton.cdiv(n, block_size),)](a, b, out, n, BLOCK_SIZE=block_size)   # 3 programs b/c triton.cdiv(10, 4) = 3

# OUTPUT
# pid [0] offsets [0 1 2 3] mask [ True  True  True  True]
# program 0: block_start = 0 * 4 = 0, so offsets = 0 + [0,1,2,3]. All four are less than 10, 
# so the mask is all True. It loads, adds, and stores elements 0-3.

# pid [1] offsets [4 5 6 7] mask [ True  True  True  True]
# program 1: block_start = 1 * 4 = 4, so offsets = 4 + [0,1,2,3] = [4 5 6 7]. 
# loads, adds, and stores elements 4-7

# pid [2] offsets [ 8  9 10 11] mask [ True  True False False]
# program 2: block_start = 2 * 4 = 8, so offsets = 8 + [0,1,2,3] = [8 9 10 11]
# 10 and 11 are out of bounds since valid indices are 0-9, so masks = false for those
