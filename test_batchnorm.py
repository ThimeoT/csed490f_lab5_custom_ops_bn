import torch
import triton
from custom_ops_bn.ops import batchnorm

def batchnorm_cuda(x, g, b, rm, rv, training):
    """Wrapper for CUDA backend batchnorm"""
    return batchnorm(x, g, b, rm, rv, training, backend='cuda')

def batchnorm_triton(x, g, b, rm, rv, training):
    """Wrapper for Triton backend batchnorm"""
    return batchnorm(x, g, b, rm, rv, training, backend='triton')

def test_correctness(N=32, C=64, H=56, W=56, training=True):
    """Test correctness of Triton implementation"""
    print(f"\n{'='*70}")
    print(f"Correctness Test: N={N}, C={C}, H={H}, W={W}, Training={training}")
    print(f"{'='*70}")

    device = 'cuda'

    # Create test tensors
    input_test = torch.randn(N, C, H, W, device=device)
    gamma_test = torch.randn(C, device=device)
    beta_test = torch.randn(C, device=device)
    rm_test = torch.randn(C, device=device)
    rv_test = torch.abs(torch.randn(C, device=device))

    # PyTorch reference
    bn_pytorch = torch.nn.BatchNorm2d(C, momentum=0.1, eps=1e-5).to(device)
    bn_pytorch.weight.data = gamma_test.clone()
    bn_pytorch.bias.data = beta_test.clone()
    bn_pytorch.running_mean.data = rm_test.clone()
    bn_pytorch.running_var.data = rv_test.clone()
    bn_pytorch.train() if training else bn_pytorch.eval()

    with torch.no_grad():
        out_pytorch = bn_pytorch(input_test.clone())

        # Test Triton implementation
        triton_pass = False
        try:
            out_triton = batchnorm(
                input_test.clone(),
                gamma_test.clone(),
                beta_test.clone(),
                rm_test.clone(),
                rv_test.clone(),
                training=training,
                backend='triton'
            )
            diff_triton = torch.max(torch.abs(out_pytorch - out_triton)).item()
            triton_pass = diff_triton < 1e-3

            print(f"\n   Triton max diff: {diff_triton:.2e}")
            print(f"   Triton: {'✓ PASSED' if triton_pass else '✗ FAILED'}")

        except NotImplementedError:
            print(f"\n   Triton: ⚠ NOT IMPLEMENTED (TODO)")

        except Exception as e:
            print(f"\n   Triton: ✗ ERROR - {type(e).__name__}: {str(e)[:80]}")

    if not training:
        # The reference backward in batchnorm.cu assumes training mode, so the
        # gradient check below only applies there.
        return triton_pass

    return triton_pass and test_backward_correctness(
        input_test, gamma_test, beta_test, rm_test, rv_test, bn_pytorch)


def test_backward_correctness(input_test, gamma_test, beta_test,
                              rm_test, rv_test, bn_pytorch):
    """Test correctness of the Triton *backward* pass.

    The forward check above runs under torch.no_grad(), so it says nothing
    about batchnorm_backward_kernel. This compares all three gradients
    (grad_input, grad_gamma, grad_beta) against torch.nn.BatchNorm2d autograd.
    """
    grad_output = torch.randn_like(input_test)

    # PyTorch reference gradients
    x_ref = input_test.clone().requires_grad_(True)
    bn_pytorch.zero_grad(set_to_none=True)
    bn_pytorch.running_mean.data = rm_test.clone()
    bn_pytorch.running_var.data = rv_test.clone()
    out_ref = bn_pytorch(x_ref)
    out_ref.backward(grad_output)

    try:
        x_tri = input_test.clone().requires_grad_(True)
        g_tri = gamma_test.clone().requires_grad_(True)
        b_tri = beta_test.clone().requires_grad_(True)
        out_tri = batchnorm(x_tri, g_tri, b_tri,
                            rm_test.clone(), rv_test.clone(),
                            training=True, backend='triton')
        out_tri.backward(grad_output)
    except NotImplementedError:
        print(f"   Triton backward: ⚠ NOT IMPLEMENTED (TODO)")
        return False
    except Exception as e:
        print(f"   Triton backward: ✗ ERROR - {type(e).__name__}: {str(e)[:80]}")
        return False

    def rel_diff(got, ref):
        """grad_gamma/grad_beta are sums over N*H*W terms, so they grow with the
        input size; compare them relative to their own magnitude."""
        denom = max(ref.abs().max().item(), 1e-3)
        return (got - ref).abs().max().item() / denom

    checks = [
        ("grad_input", rel_diff(x_tri.grad, x_ref.grad)),
        ("grad_gamma", rel_diff(g_tri.grad, bn_pytorch.weight.grad)),
        ("grad_beta", rel_diff(b_tri.grad, bn_pytorch.bias.grad)),
    ]
    passed = all(d < 1e-3 for _, d in checks)
    print(f"\n   Triton backward (relative diff vs torch.nn.BatchNorm2d):")
    for name, d in checks:
        print(f"      {name:<12} {d:.2e}  {'✓' if d < 1e-3 else '✗'}")
    print(f"   Triton backward: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def benchmark_forward_backward(func, input, gamma, beta, running_mean, running_var,
                                training=True, warmup_ms=50, rep_ms=200):
    """Measure the forward and the backward kernel separately, in milliseconds.

    Two things matter for getting a number you can actually reason about:

    1. `triton.testing.do_bench` times with CUDA events and flushes the L2
       between repetitions, so every repetition reads cold memory the way the
       first call in a real training step does. Timing with time.time() around
       a synchronize adds 10-50 us of host overhead, which is a large error on
       a kernel that runs in a few hundred microseconds.

    2. The backward is measured with torch.autograd.grad(retain_graph=True)
       rather than loss.backward(). Calling output.sum().backward() would put a
       full N*C*H*W reduction inside the measurement, which is not part of the
       batch-norm backward at all.

    The median is reported, so a single scheduling hiccup does not move it.
    """
    inp = input.clone().detach().requires_grad_(True)
    g = gamma.clone().detach().requires_grad_(True)
    b = beta.clone().detach().requires_grad_(True)

    def forward():
        return func(inp, g, b, running_mean.clone(), running_var.clone(),
                    training=training)

    forward_time = triton.testing.do_bench(
        forward, warmup=warmup_ms, rep=rep_ms, return_mode='median')

    output = forward()
    grad_output = torch.randn_like(output)

    def backward():
        return torch.autograd.grad(output, (inp, g, b), grad_output,
                                   retain_graph=True)

    backward_time = triton.testing.do_bench(
        backward, warmup=warmup_ms, rep=rep_ms, return_mode='median')

    return forward_time, backward_time


def run_benchmark(N, C, H, W, training=True):
    """Run performance benchmark for specific input size"""
    print(f"\n{'='*70}")
    print(f"Performance Benchmark: N={N}, C={C}, H={H}, W={W}, Training={training}")
    print(f"{'='*70}")

    device = 'cuda'

    # Create base tensors
    input_base = torch.randn(N, C, H, W, device=device)
    gamma = torch.randn(C, device=device)
    beta = torch.randn(C, device=device)
    running_mean = torch.randn(C, device=device)
    running_var = torch.abs(torch.randn(C, device=device))

    # CUDA custom operator
    print("\n1. CUDA Custom Operator (Reference)")
    try:
        forward_time_cuda, backward_time_cuda = benchmark_forward_backward(
            batchnorm_cuda,
            input_base, gamma, beta,
            running_mean, running_var, training
        )
        print(f"   Forward:  {forward_time_cuda:.4f} ms")
        print(f"   Backward: {backward_time_cuda:.4f} ms")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return

    # Triton custom operator
    print("\n2. Triton Custom Operator (Your Implementation)")
    try:
        forward_time_triton, backward_time_triton = benchmark_forward_backward(
            batchnorm_triton,
            input_base, gamma, beta,
            running_mean, running_var, training
        )
        print(f"   Forward:  {forward_time_triton:.4f} ms")
        print(f"   Speedup vs CUDA: {forward_time_cuda / forward_time_triton:.2f}x")
        print(f"   Backward: {backward_time_triton:.4f} ms")
        print(f"   Speedup vs CUDA: {backward_time_cuda / backward_time_triton:.2f}x")
    except NotImplementedError:
        print(f"   ⚠ NOT IMPLEMENTED (TODO)")
    except Exception as e:
        print(f"   ✗ ERROR: {type(e).__name__}: {str(e)[:80]}")


def main():
    print("\n" + "="*70)
    print(" BatchNorm Custom Operators: Correctness & Performance Tests")
    print("="*70)

    # Test configurations
    configs = [
        # (8, 32, 224, 224),    
        # (32, 64, 112, 112),   
        (64, 128, 56, 56),    
    ]

    # Run correctness tests *************************************************************************
    print("\n" + "="*70)
    print(" CORRECTNESS TESTS")
    print("="*70)

    all_passed = True
    for N, C, H, W in configs:
        passed_train = test_correctness(N, C, H, W, training=True)
        passed_eval = test_correctness(N, C, H, W, training=False)
        all_passed = all_passed and passed_train and passed_eval

    print("\n" + "="*70)
    if all_passed:
        print(" ALL CORRECTNESS TESTS PASSED")
    else:
        print(" SOME CORRECTNESS TESTS FAILED")
    print("="*70)

    # Run performance benchmarks *************************************************************************
    print("\n" + "="*70)
    print(" PERFORMANCE BENCHMARKS")
    print("="*70)

    for N, C, H, W in configs:
        # Training mode
        run_benchmark(N, C, H, W, training=True)

        # Inference mode
        run_benchmark(N, C, H, W, training=False)

    print("\n" + "="*70)
    print(" Tests Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()