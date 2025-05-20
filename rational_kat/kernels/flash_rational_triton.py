import torch
import triton
import triton.language as tl
from torch import Tensor
from math import ceil

# --------------------
# Forward kernel
# --------------------
# The forward kernel computes for each element:
#   P = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5   (computed by Horner’s method)
#   Q = 1 + |b0|*|x| + |b1|*|x|^2 + |b2|*|x|^3 + |b3|*|x|^4
#   result = P / Q
#
# Each “group” uses 6 coefficients from a and 4 coefficients from b.
#
# We assume the following inputs:
#   x_ptr: pointer to input tensor (flattened, size = B*L*D)
#   a_ptr: pointer to numerator coefficients (per–group, groups = group count)
#   b_ptr: pointer to denominator coefficients (per–group)
#   result_ptr: pointer to output tensor (flattened)
#   x_size: total number of elements
#   D: size of the last dimension
#   D_per_group: D divided by the number of groups
#

@triton.jit
def rational_fwd_kernel(
    x_ptr, a_ptr, b_ptr, result_ptr,
    D, x_size,
    BLOCK_SIZE: tl.constexpr,
    D_per_group: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < x_size

    # Load input elements.
    x_val = tl.load(x_ptr + offs, mask=mask)

    # Determine d index and group index.
    d_index = offs % D
    g_index = d_index // D_per_group

    # Compute coefficient offsets.
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load numerator coefficients.
    s_a0 = tl.load(a_ptr + a_offset + 0)
    s_a1 = tl.load(a_ptr + a_offset + 1)
    s_a2 = tl.load(a_ptr + a_offset + 2)
    s_a3 = tl.load(a_ptr + a_offset + 3)
    s_a4 = tl.load(a_ptr + a_offset + 4)
    s_a5 = tl.load(a_ptr + a_offset + 5)

    # Load denominator coefficients (using absolute value).
    s_b0 = tl.abs(tl.load(b_ptr + b_offset + 0))
    s_b1 = tl.abs(tl.load(b_ptr + b_offset + 1))
    s_b2 = tl.abs(tl.load(b_ptr + b_offset + 2))
    s_b3 = tl.abs(tl.load(b_ptr + b_offset + 3))

    abs_x = tl.abs(x_val)

    # Compute numerator polynomial P(x) via Horner's method.
    P = s_a5
    P = tl.fma(P, x_val, s_a4)
    P = tl.fma(P, x_val, s_a3)
    P = tl.fma(P, x_val, s_a2)
    P = tl.fma(P, x_val, s_a1)
    P = tl.fma(P, x_val, s_a0)

    # Compute denominator polynomial Q(x).
    Q = s_b3
    Q = tl.fma(Q, abs_x, s_b2)
    Q = tl.fma(Q, abs_x, s_b1)
    Q = tl.fma(Q, abs_x, s_b0)
    Q = tl.fma(Q, abs_x, 1.0)

    tl.store(result_ptr + offs, P / Q, mask=mask)

def rational_fwd_triton(x, n, d):
    D = x.shape[-1]
    x_size = x.numel()

    group = max(n.shape[0], d.shape[0])
    result = torch.empty_like(x)
    BLOCK_SIZE = 128
    num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    rational_fwd_kernel[(num_blocks,)](
        x, n, d, result,
        D, x_size, 
        BLOCK_SIZE=BLOCK_SIZE,
        D_per_group=D // group
    )

    return result

# --------------------
# Backward kernel
# --------------------
# The backward kernel computes gradients with respect to the input x and the coefficients.
# For each element it computes:
#
#   xp = x
#   axp = |x|
#   P = a0 + a1*x + a2*x^2 + ... + a5*x^5
#   Q = 1 + |b0|*axp + |b1|*axp^2 + |b2|*axp^3 + |b3|*axp^4
#   R = a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 + 5*a5*x^4
#   S = sign(x) * (|b0| + 2*|b1|*axp + 3*|b2|*axp^2 + 4*|b3|*axp^3)
#
# and then:
#   d_x = (R/Q + S * (-P/(Q^2))) * grad_o
#
# It also computes per–coefficient gradients:
#
#   d_a[0] = grad_o/Q,  d_a[i] = (x^i * grad_o)/Q, for i = 1,...,5
#   d_b[i] = (-P/(Q^2)) * sign(b[i]) * (axp^(i+1)) * grad_o, for i = 0,...,3
#
# The results for d_a and d_b are accumulated via atomic adds.

@triton.jit
def rational_bwd_kernel(
    # pointers
    grad_output_ptr, x_ptr,
    a_ptr, b_ptr,
    d_x_ptr, d_a_ptr, d_b_ptr,
    # scalars
    B: tl.constexpr, 
    din: tl.constexpr,
    # original group_size = number of channels per group
    group_size: tl.constexpr,
    # M-tiling factors
    BLOCK_M_OUT: tl.constexpr,
    BLOCK_M_IN:  tl.constexpr,
    # N-tiling factors (within each group)
    GROUP_OUT: tl.constexpr,
    GROUP_IN:  tl.constexpr,
):
    # polynomial orders
    M = 6
    N = 4

    batch_tile = tl.program_id(0)
    group      = tl.program_id(1)

    # base row offset for this big tile
    base_row = batch_tile * (BLOCK_M_OUT * BLOCK_M_IN)
    # base column offset into this group
    base_col = group * group_size

    # load the 1×M and 1×N diagonal coefficients once
    a0 = tl.load(a_ptr + group*M + 0);  a1 = tl.load(a_ptr + group*M + 1)
    a2 = tl.load(a_ptr + group*M + 2);  a3 = tl.load(a_ptr + group*M + 3)
    a4 = tl.load(a_ptr + group*M + 4);  a5 = tl.load(a_ptr + group*M + 5)
    b0 = tl.load(b_ptr + group*N + 0);  b1 = tl.load(b_ptr + group*N + 1)
    b2 = tl.load(b_ptr + group*N + 2);  b3 = tl.load(b_ptr + group*N + 3)
    # abs‐versions
    b0a, b1a, b2a, b3a = tl.abs(b0), tl.abs(b1), tl.abs(b2), tl.abs(b3)

    # accumulators for coef‐grads
    da0_sum = tl.zeros([], dtype=tl.float32)
    da1_sum = tl.zeros([], dtype=tl.float32)
    da2_sum = tl.zeros([], dtype=tl.float32)
    da3_sum = tl.zeros([], dtype=tl.float32)
    da4_sum = tl.zeros([], dtype=tl.float32)
    da5_sum = tl.zeros([], dtype=tl.float32)
    db0_sum = tl.zeros([], dtype=tl.float32)
    db1_sum = tl.zeros([], dtype=tl.float32)
    db2_sum = tl.zeros([], dtype=tl.float32)
    db3_sum = tl.zeros([], dtype=tl.float32)

    for mo in range(BLOCK_M_OUT):
        # compute the row‐indices
        row_idx  = base_row + mo * BLOCK_M_IN + tl.arange(0, BLOCK_M_IN)
        row_mask = row_idx < B

        for no in range(GROUP_OUT):
            # compute the column‐indices
            col_idx  = base_col + no * GROUP_IN + tl.arange(0, GROUP_IN)
            col_mask = (col_idx >= base_col) & (col_idx < base_col + group_size)

            # build 2D masks & offsets
            mask2d = row_mask[:, None] & col_mask[None, :]
            offs   = row_idx[:, None] * din + col_idx[None, :]

            # load tiles of x and grad_out
            x_tile    = tl.load(x_ptr    + offs, mask=mask2d, other=0.0)
            grad_tile = tl.load(grad_output_ptr + offs, mask=mask2d, other=0.0)

            # forward‐polynomial powers
            xp   = x_tile
            xp2  = xp*xp; xp3 = xp2*xp; xp4 = xp3*xp; xp5 = xp4*xp
            axp   = tl.abs(xp)
            axp2  = axp*axp; axp3 = axp2*axp; axp4 = axp3*axp

            # evaluate P, Q
            P = a0 + a1*xp + a2*xp2 + a3*xp3 + a4*xp4 + a5*xp5
            Q = 1.0 + b0a*axp + b1a*axp2 + b2a*axp3 + b3a*axp4

            # derivatives R, S, mpq2
            R      = a1 + 2*a2*xp + 3*a3*xp2 + 4*a4*xp3 + 5*a5*xp4
            sign_x = tl.where(x_tile < 0, -1.0, 1.0)
            S      = sign_x * (b0a + 2*b1a*axp + 3*b2a*axp2 + 4*b3a*axp3)
            mpq2   = -P / (Q * Q)

            # grad w.r.t x
            dx = (R/Q + S*mpq2) * grad_tile
            tl.store(d_x_ptr + offs, dx, mask=mask2d)

            # grad w.r.t a_i
            da0 =        grad_tile / Q
            da1 = xp    * grad_tile / Q
            da2 = xp2   * grad_tile / Q
            da3 = xp3   * grad_tile / Q
            da4 = xp4   * grad_tile / Q
            da5 = xp5   * grad_tile / Q

            # grad w.r.t b_i (keep original sign of b_i)
            sb0 = tl.where(b0 < 0, -1.0, 1.0)
            sb1 = tl.where(b1 < 0, -1.0, 1.0)
            sb2 = tl.where(b2 < 0, -1.0, 1.0)
            sb3 = tl.where(b3 < 0, -1.0, 1.0)
            db0 = mpq2 * sb0 * axp  * grad_tile
            db1 = mpq2 * sb1 * axp2 * grad_tile
            db2 = mpq2 * sb2 * axp3 * grad_tile
            db3 = mpq2 * sb3 * axp4 * grad_tile

            # sum over this mini‐tile
            da0_sum += tl.sum(da0)
            da1_sum += tl.sum(da1)
            da2_sum += tl.sum(da2)
            da3_sum += tl.sum(da3)
            da4_sum += tl.sum(da4)
            da5_sum += tl.sum(da5)
            db0_sum += tl.sum(db0)
            db1_sum += tl.sum(db1)
            db2_sum += tl.sum(db2)
            db3_sum += tl.sum(db3)

    # one single atomic add per coefficient
    tl.atomic_add(d_a_ptr + group*M + 0, da0_sum)
    tl.atomic_add(d_a_ptr + group*M + 1, da1_sum)
    tl.atomic_add(d_a_ptr + group*M + 2, da2_sum)
    tl.atomic_add(d_a_ptr + group*M + 3, da3_sum)
    tl.atomic_add(d_a_ptr + group*M + 4, da4_sum)
    tl.atomic_add(d_a_ptr + group*M + 5, da5_sum)

    tl.atomic_add(d_b_ptr + group*N + 0, db0_sum)
    tl.atomic_add(d_b_ptr + group*N + 1, db1_sum)
    tl.atomic_add(d_b_ptr + group*N + 2, db2_sum)
    tl.atomic_add(d_b_ptr + group*N + 3, db3_sum)


def rational_bwd_triton(grad_output, x, n, d):
    """
    grad_output: gradient of loss w.r.t. output, shape [B, L, din]
    x: input tensor, shape [B, L, din]
    n: numerator coefficients, shape [din, din, 6]  (dout==din, so we use the diagonal)
    d: denominator coefficients, shape [din, din, 4]
    
    Returns gradients with respect to x, n and d.
    """
    B_orig, L, din = x.shape
    B = B_orig * L  # flattened batch dimension
    # Reshape x and grad_output to [B, din]
    num_groups = max(n.shape[0], d.shape[0])
    x_flat = x.view(B, din)
    grad_flat = grad_output.view(B, din)

    # Allocate gradient outputs.
    d_x = torch.empty_like(x_flat, dtype=torch.float32)
    d_n = torch.zeros_like(n, dtype=torch.float32)
    d_d = torch.zeros_like(d, dtype=torch.float32)

    # We use BLOCK_M tiling only along the batch dimension.
    group_size = din // num_groups
    BLOCK_M_OUT, BLOCK_M_IN = 1, 64
    GROUP_IN = 64 
    GROUP_OUT = (group_size + GROUP_IN - 1) // GROUP_IN
    grid0 = (B + BLOCK_M_OUT*BLOCK_M_IN - 1) // (BLOCK_M_OUT*BLOCK_M_IN)
    grid1 = num_groups

    rational_bwd_kernel[ (grid0, grid1) ](
        grad_flat, x_flat, n, d, d_x, d_n, d_d,
        B, din,
        group_size,
        BLOCK_M_OUT, BLOCK_M_IN,
        GROUP_OUT,   GROUP_IN
    )

    d_x = d_x.view(B_orig, L, din)
    return d_x, d_n, d_d



class FlashRationalTriton1DGroup(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx: torch.autograd.Function, 
                input: Tensor, 
                weight_numerator: Tensor, 
                weight_denominator: Tensor) -> Tensor:
        """
        Forward pass of the rational function computed with Triton kernels.
        
        Args:
            ctx: The context object for storing information for the backward pass.
            input (Tensor): Input tensor.
            weight_numerator (Tensor): Weights for the numerator polynomial.
            weight_denominator (Tensor): Weights for the denominator polynomial.
            group (int): The group number (non-differentiable).
        
        Returns:
            Tensor: Output tensor resulting from applying the rational function.
        """
        # Save tensors required for backward pass.
        ctx.save_for_backward(input, weight_numerator, weight_denominator)

        # Compute the forward pass using a Triton kernel.
        output = rational_fwd_triton(input, weight_numerator, weight_denominator)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx: torch.autograd.Function, grad_output: Tensor):
        """
        Backward pass of the rational function computed with Triton kernels.
        
        Args:
            ctx: The context object with saved tensors.
            grad_output (Tensor): Gradient of the loss with respect to the output.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, None]:
                - Gradient with respect to the input.
                - Gradient with respect to weight_numerator.
                - Gradient with respect to weight_denominator.
                - None for the non-differentiable 'group' parameter.
        """
        # Retrieve saved tensors and the group number.
        input, weight_numerator, weight_denominator = ctx.saved_tensors

        # Compute gradients using a Triton backward kernel.
        d_input, d_weight_numerator, d_weight_denominator = rational_bwd_triton(
            grad_output, input, weight_numerator, weight_denominator
        )

        # Return gradients. None is returned for 'group' as it is non-differentiable.
        return d_input, d_weight_numerator, d_weight_denominator