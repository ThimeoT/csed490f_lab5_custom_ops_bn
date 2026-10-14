#include <torch/all.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

namespace custom_ops_bn {

/**
 * CUDA kernel for batch normalization forward pass (training mode)
 *
 * Architecture:
 *   - One block per channel (blockIdx.x = channel index)
 *   - Multiple threads per block for parallel reduction
 *
 * Algorithm (Two-pass algorithm):
 *   1. Pass 1: Compute mean using parallel reduction
 *   2. Pass 2: Compute variance using the mean
 *   3. Update running statistics (EMA)
 *   4. Pass 3: Normalize input and apply affine transformation
 *
 *
 * Memory layout: NCHW (batch, channel, height, width)
 * Index calculation: idx = n * C * spatial_dim + c * spatial_dim + s
 *
 */
__global__ void batchnorm_forward_training_kernel(
    const float* input,          // Input tensor [N, C, H, W]
    const float* gamma,          // Scale parameter [C]
    const float* beta,           // Shift parameter [C]
    float* output,               // Output tensor [N, C, H, W]
    float* save_mean,            // Saved mean for backward [C]
    float* save_invstd,          // Saved inverse std for backward [C]
    float* running_mean,         // Running mean (updated in-place) [C]
    float* running_var,          // Running variance (updated in-place) [C]
    float momentum,              // Momentum for running stats update
    float eps,                   // Epsilon for numerical stability
    int N,                       // Batch size
    int C,                       // Number of channels
    int spatial_dim) {           // H * W

    extern __shared__ float shared_data[];

    const int c = blockIdx.x;              // Channel index
    const int tid = threadIdx.x;           // Thread index within block
    const int blockSize = blockDim.x;      // Number of threads per block

    if (c >= C) return;

    float* partial_sum = shared_data;  // Shared memory for reduction

    // Compute work distribution for this thread
    const int total_elements = N * spatial_dim;
    const int elements_per_thread = (total_elements + blockSize - 1) / blockSize;
    const int start = tid * elements_per_thread;
    const int end = min(start + elements_per_thread, total_elements);

    // ========== PASS 1: Compute mean ==========
    partial_sum[tid] = 0.0f;

    // Each thread computes partial sum
    for (int i = start; i < end; i++) {
        const int n = i / spatial_dim;                              // Batch index
        const int s = i % spatial_dim;                              // Spatial index
        const int idx = n * C * spatial_dim + c * spatial_dim + s;  // Linear index
        partial_sum[tid] += input[idx];
    }

    __syncthreads();

    // Parallel reduction in shared memory (tree reduction)
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 computes final mean
    if (tid == 0) {
        const float mean = partial_sum[0] / total_elements;
        save_mean[c] = mean;
    }

    __syncthreads();

    const float mean_val = save_mean[c];

    // ========== PASS 2: Compute variance ==========
    partial_sum[tid] = 0.0f;

    // Each thread computes partial sum of squared differences
    for (int i = start; i < end; i++) {
        const int n = i / spatial_dim;
        const int s = i % spatial_dim;
        const int idx = n * C * spatial_dim + c * spatial_dim + s;
        const float diff = input[idx] - mean_val;
        partial_sum[tid] += diff * diff;
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 computes final variance and updates running stats
    if (tid == 0) {
        const float variance = partial_sum[0] / total_elements;
        const float invstd = rsqrtf(variance + eps);  // 1 / sqrt(var + eps)

        // Save statistics for backward pass
        save_invstd[c] = invstd;

        // Update running statistics using exponential moving average
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean_val;
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * variance;
    }

    __syncthreads();

    // ========== PASS 3: Normalize and scale ==========
    const float invstd_val = save_invstd[c];
    const float gamma_val = gamma[c];
    const float beta_val = beta[c];

    for (int i = start; i < end; i++) {
        const int n = i / spatial_dim;
        const int s = i % spatial_dim;
        const int idx = n * C * spatial_dim + c * spatial_dim + s;
        const float x_normalized = (input[idx] - mean_val) * invstd_val;
        output[idx] = x_normalized * gamma_val + beta_val;  // y = γ * x_norm + β
    }
}

/**
 * CUDA kernel for batch normalization forward pass (inference mode)
 *
 * Architecture:
 *   - One thread per element (standard grid-stride loop)
 *
 * Algorithm:
 *   - Uses precomputed running statistics (no mean/var calculation)
 *   - Applies normalization and affine transformation
 *
 */
__global__ void batchnorm_forward_inference_kernel(
    const float* input,          // Input tensor [N, C, H, W]
    const float* gamma,          // Scale parameter [C]
    const float* beta,           // Shift parameter [C]
    const float* mean,           // Running mean [C]
    const float* variance,       // Running variance [C]
    float* output,               // Output tensor [N, C, H, W]
    float* save_invstd,          // Saved inverse std (for backward) [C]
    float eps,                   // Epsilon for numerical stability
    int N, int C, int spatial_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C * spatial_dim;

    if (idx >= total_elements) return;

    // Decompose linear index to (n, c, s)
    const int s = idx % spatial_dim;
    const int c = (idx / spatial_dim) % C;
    const int n = idx / (C * spatial_dim);

    const float mean_val = mean[c];
    const float var_val = variance[c];
    const float invstd = rsqrtf(var_val + eps);

    // Save invstd for backward pass (only first thread per channel writes)
    if (idx == c * spatial_dim) {
        save_invstd[c] = invstd;
    }

    // Normalize and apply affine transformation
    const float x_normalized = (input[idx] - mean_val) * invstd;
    output[idx] = x_normalized * gamma[c] + beta[c];
}

/**
 * CUDA kernel for batch normalization backward pass
 *
 * Architecture:
 *   - One block per channel (blockIdx.x = channel index)
 *   - Multiple threads per block for parallel reduction
 *
 * Algorithm:
 *   1. Step 1: Compute partial sums for gradients
 *   2. Step 2: Parallel reduction to aggregate results
 *   3. Step 3: Store parameter gradients (gamma, beta)
 *   4. Step 4: Compute input gradient using chain rule
 *
 */
__global__ void batchnorm_backward_kernel(
    const float* grad_output,     // Gradient from next layer [N, C, H, W]
    const float* input,           // Original input [N, C, H, W]
    const float* gamma,           // Scale parameter [C]
    const float* save_mean,       // Saved mean from forward [C]
    const float* save_invstd,     // Saved inverse std from forward [C]
    float* grad_input,            // Gradient w.r.t. input [N, C, H, W]
    float* grad_gamma,            // Gradient w.r.t. gamma [C]
    float* grad_beta,             // Gradient w.r.t. beta [C]
    int N, int C, int spatial_dim) {

    extern __shared__ float shared_grad[];

    const int tid = threadIdx.x;
    const int c = blockIdx.x;

    if (c >= C) return;

    // Partition shared memory for four reduction operations
    float* grad_gamma_sum = shared_grad;
    float* grad_beta_sum = &shared_grad[blockDim.x];
    float* sum_dy = &shared_grad[2 * blockDim.x];
    float* sum_dy_x_normalized = &shared_grad[3 * blockDim.x];

    // Initialize shared memory
    grad_gamma_sum[tid] = 0.0f;
    grad_beta_sum[tid] = 0.0f;
    sum_dy[tid] = 0.0f;
    sum_dy_x_normalized[tid] = 0.0f;

    const float mean_val = save_mean[c];
    const float invstd_val = save_invstd[c];
    const float gamma_val = gamma[c];

    // Compute work distribution
    const int total_elements = N * spatial_dim;
    const int elements_per_thread = (total_elements + blockDim.x - 1) / blockDim.x;
    const int start = tid * elements_per_thread;
    const int end = min(start + elements_per_thread, total_elements);

    // Step 1: Compute partial sums for gradients
    for (int i = start; i < end; i++) {
        const int n = i / spatial_dim;
        const int s = i % spatial_dim;
        const int idx = n * C * spatial_dim + c * spatial_dim + s;

        const float grad_out = grad_output[idx];
        const float x_normalized = (input[idx] - mean_val) * invstd_val;

        grad_beta_sum[tid] += grad_out;
        grad_gamma_sum[tid] += grad_out * x_normalized;
        sum_dy[tid] += grad_out;
        sum_dy_x_normalized[tid] += grad_out * x_normalized;
    }

    __syncthreads();

    // Step 2: Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            grad_gamma_sum[tid] += grad_gamma_sum[tid + stride];
            grad_beta_sum[tid] += grad_beta_sum[tid + stride];
            sum_dy[tid] += sum_dy[tid + stride];
            sum_dy_x_normalized[tid] += sum_dy_x_normalized[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Thread 0 writes parameter gradients
    if (tid == 0) {
        grad_gamma[c] = grad_gamma_sum[0];
        grad_beta[c] = grad_beta_sum[0];
    }

    __syncthreads();

    // Step 4: Compute input gradient using chain rule
    const float M = static_cast<float>(total_elements);
    const float sum_dy_val = sum_dy[0];
    const float sum_dy_x_normalized_val = sum_dy_x_normalized[0];

    for (int i = start; i < end; i++) {
        const int n = i / spatial_dim;
        const int s = i % spatial_dim;
        const int idx = n * C * spatial_dim + c * spatial_dim + s;

        const float dy_val = grad_output[idx];
        const float x_normalized = (input[idx] - mean_val) * invstd_val;

        // Chain rule: grad_input = grad_output * d(output)/d(input)
        grad_input[idx] = (gamma_val * invstd_val / M) *
                         (M * dy_val - sum_dy_val - x_normalized * sum_dy_x_normalized_val);
    }
}

// CUDA implementation wrapper
std::vector<at::Tensor> batchnorm_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps) {

    auto N = input.size(0);
    auto C = input.size(1);
    auto spatial_dim = input.numel() / (N * C);

    TORCH_CHECK(input.is_cuda());
    TORCH_CHECK(gamma.is_cuda());
    TORCH_CHECK(beta.is_cuda());
    TORCH_CHECK(input.is_contiguous());

    at::Tensor output = torch::empty_like(input);
    at::Tensor save_mean = torch::empty({C}, input.options());
    at::Tensor save_invstd = torch::empty({C}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* beta_ptr = beta.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* save_mean_ptr = save_mean.data_ptr<float>();
    float* save_invstd_ptr = save_invstd.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (training) {
        // Use parallelized training kernel
        float* running_mean_ptr = running_mean.data_ptr<float>();
        float* running_var_ptr = running_var.data_ptr<float>();

        int blocks = C;  // One block per channel
        int threads = 256;  // Use multiple threads for parallel processing
        int shared_mem = threads * sizeof(float);  // For partial_sum (reused for both passes)

        batchnorm_forward_training_kernel<<<blocks, threads, shared_mem, stream>>>(
            input_ptr, gamma_ptr, beta_ptr,
            output_ptr, save_mean_ptr, save_invstd_ptr,
            running_mean_ptr, running_var_ptr,
            momentum, eps, N, C, spatial_dim
        );
    } else {
        // Use running statistics for inference
        const float* running_mean_ptr = running_mean.data_ptr<float>();
        const float* running_var_ptr = running_var.data_ptr<float>();

        save_mean.copy_(running_mean);

        int total_elements = N * C * spatial_dim;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        batchnorm_forward_inference_kernel<<<blocks, threads, 0, stream>>>(
            input_ptr, gamma_ptr, beta_ptr,
            running_mean_ptr, running_var_ptr,
            output_ptr, save_invstd_ptr,
            eps, N, C, spatial_dim
        );
    }

    return {output, save_mean, save_invstd};
}

// Note: This backward implementation assumes training mode.
std::vector<at::Tensor> batchnorm_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& gamma,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd) {

    auto N = input.size(0);
    auto C = input.size(1);
    auto spatial_dim = input.numel() / (N * C);

    // Ensure inputs are contiguous
    auto grad_output_contig = grad_output.contiguous();
    auto input_contig = input.contiguous();

    at::Tensor grad_input = torch::empty_like(input);
    at::Tensor grad_gamma = torch::zeros({C}, gamma.options());
    at::Tensor grad_beta = torch::zeros({C}, gamma.options());

    const float* grad_output_ptr = grad_output_contig.data_ptr<float>();
    const float* input_ptr = input_contig.data_ptr<float>();
    const float* gamma_ptr = gamma.data_ptr<float>();
    const float* save_mean_ptr = save_mean.data_ptr<float>();
    const float* save_invstd_ptr = save_invstd.data_ptr<float>();
    float* grad_input_ptr = grad_input.data_ptr<float>();
    float* grad_gamma_ptr = grad_gamma.data_ptr<float>();
    float* grad_beta_ptr = grad_beta.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;
    int blocks = C;
    int shared_mem = threads * 4 * sizeof(float);

    batchnorm_backward_kernel<<<blocks, threads, shared_mem, stream>>>(
        grad_output_ptr, input_ptr, gamma_ptr,
        save_mean_ptr, save_invstd_ptr,
        grad_input_ptr, grad_gamma_ptr, grad_beta_ptr,
        N, C, spatial_dim
    );

    return {grad_input, grad_gamma, grad_beta};
}

} // namespace custom_ops_bn