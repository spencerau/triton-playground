import torch

import triton
import triton.language as tl
from triton.runtime import driver

# torch.cuda.set_device(1)
DEVICE = triton.runtime.driver.active.get_active_torch_device()

# cant really set gpu via code for some reason so use env var
# HIP_VISIBLE_DEVICES=1 python src/02_fused_softmax/softmax_temperature.py

props = torch.cuda.get_device_properties(DEVICE)
print(props.name)


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


@triton.jit
def softmax_kernel(output_ptr, input_ptr, 
                   input_row_stride, output_row_stride, 
                   input_col_stride, output_col_stride,
                   n_rows, n_cols,
                   temperature,
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
        numerator = tl.exp(row_minus_max / temperature)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # write output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets * output_col_stride
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
def softmax_temp(x, temperature):
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
                                   x.stride(1), y.stride(1),
                                   n_rows, n_cols,
                                   temperature,
                                   BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages,
                                   num_warps=num_warps,
                                   grid=(1, ))

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    # print("Block Size:", BLOCK_SIZE, "; Size SMEM:", size_smem)

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

    # limit occupancy by shared memory size
    if (size_smem > 0):
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
    # else use register basd value

    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    # create a number of persistent programs
    kernel[(num_programs, 1, 1)](y, x, 
                                 x.stride(0), y.stride(0),
                                 x.stride(1), y.stride(1),
                                 n_rows, n_cols,
                                 temperature,
                                 BLOCK_SIZE,
                                 num_stages)

    return y


torch.manual_seed(0)
for shape in [(1, 5), (4, 10), (128, 781), (1000, 1024)]:
    for T in [0.5, 1.0, 2.0, 10.0]:
        x = torch.randn(shape, device=DEVICE)
        out = softmax_temp(x, T)
        expected = torch.softmax(x / T, dim=1)
        ok = torch.allclose(out, expected, atol=1e-6)
        print(f"shape={shape} T={T}: {'OK' if ok else 'MISMATCH'}")

