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


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


# row wise softmax using PyTorch - SLOW
def naive_softmax(x):
    """
    Compute row-wise softmax of X using native PyTorch

    subtract maximum element in order to avoid overflows. Softmax is invariant to this shift
    """
    # read MN elements, write M elements
    x_max = x.max(dim=1)[0]

    # read MN + M elements, write MN elements
    z = x - x_max[:, None]

    # read MN elements, write MN elements
    numerator = torch.exp(z)

    # read MN elements, write M elements
    denominator = numerator.sum(dim=1)

    # read MN + M elements, write MN elements
    ret = numerator / denominator[:, None]

    # in total: read 5MN + 2M elements, wrote 3MN + 2M elements
    return ret


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

# meta-arguments
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# function/code that prepares and launches that kernel 
def softmax(x):
    n_rows, n_cols = x.shape

    # block size of each loop iteration is the smallest power of 2 greater than num of cols in 'x'
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # num of software pipelining stages
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel = softmax_kernel.warmup(y, x,
                                   x.stride(0), y.stride(0),
                                   n_rows, n_cols,
                                   BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages,
                                   num_warps=num_warps,
                                   grid=(1, ))

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2
        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

    occupancy = min(occupancy, SIZE_SMEM // size_smem)

    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    # create a number of persistent programs
    kernel[(num_programs, 1, 1)](y, x, 
                                 x.stride(0), y.stride(0),
                                 n_rows, n_cols,
                                 BLOCK_SIZE,
                                 num_stages)

    return y


# torch.manual_seed(0)
# x = torch.randn(1823, 781, device=DEVICE)
# y_triton = softmax(x)
# y_torch = torch.softmax(x, axis=1)
# assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=['N'],  # argument names to use as an x-axis for the plot
#         x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
#         line_arg='provider',  # argument name whose value corresponds to a different line in the plot
#         line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
#         line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
#         styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
#         ylabel="GB/s",  # label name for the y-axis
#         plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
#         args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
#     ))
# def benchmark(M, N, provider):
#     x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
#     if provider == 'torch':
#         ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
#     if provider == 'triton':
#         ms = triton.testing.do_bench(lambda: softmax(x))
#     if provider == 'naive_softmax':
#         ms = triton.testing.do_bench(lambda: naive_softmax(x))
#     gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
#     return gbps(ms)


# headless cluster: no display for show_plots, so save the png (and csv) next to this file instead
# benchmark.run(print_data=True, save_path="src/02_fused_softmax/bench")

# print("\n--- persistent loop demo ---")
# n_rows, n_cols = 10, 5
# num_programs = 3   # try 1, 10, 20
# x = torch.randn(n_rows, n_cols, device=DEVICE)
# y = torch.empty_like(x)
# softmax_kernel[(num_programs,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
#                                 BLOCK_SIZE=8, num_stages=1)
# print("matches torch:", torch.allclose(y, torch.softmax(x, dim=1)))

# --- OUTPUT ---
# pid [0] row 0
# pid [0] row 3
# pid [0] row 6
# pid [0] row 9
# pid [1] row 1
# pid [1] row 4
# pid [1] row 7
# pid [2] row 2
# pid [2] row 5
# pid [2] row 8
# matches torch: True

# persistent loop has 3 programs sharing 10 rows by taking turns
# basically we have 3 programs, pid0, pid1, pid2. pid0 starts at row 0, pid1 starts at row 1, 
# and pid2 starts at row 2. then they get incremented by an offset to the "next row", so pid0 
# goes to row 0 + num_procs = row 3, pid1 goes to row 1 + num_proces = 4, and so on

# pid 0: 0 -> 3 -> 6 -> 9
# pid 1: 1 -> 4 -> 7
# pid 2: 2 -> 5 -> 8


def run_softmax_kernel_demo(values, num_programs=1, block_size=None):
    x = torch.tensor(values, dtype=torch.float32, device=DEVICE)
    n_rows, n_cols = x.shape
    if block_size is None:
        block_size = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(num_programs,)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,
                                    BLOCK_SIZE=block_size, num_stages=1)
    expected = torch.softmax(x, dim=1)
    print("triton:", y)
    print("torch: ", expected)
    print("match: ", torch.allclose(y, expected))
    return y


run_softmax_kernel_demo([[1., 2., 3., 4., 5.]])

# run_softmax_kernel_demo([[1., 2., 3., 4., 5.], [5., 5., 5., 5., 5.]])   # two rows, one program loops over both
# run_softmax_kernel_demo([[100., 200., 300.]])                            # overflow test for the max-subtraction experiment
# run_softmax_kernel_demo([[1., 2., 3.]], block_size=16)                   # lots of padding

