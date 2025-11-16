import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized 3D noise addition kernel
__global__ void add_noise_3d_kernel(
    const float* __restrict__ x_start,
    const float* __restrict__ noise,
    float* __restrict__ output,
    const float sqrt_alpha_cumprod,
    const float sqrt_one_minus_alpha_cumprod,
    const int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sqrt_alpha_cumprod * x_start[idx] + 
                      sqrt_one_minus_alpha_cumprod * noise[idx];
    }
}

// Launch kernel function
torch::Tensor add_noise_3d_cuda(
    torch::Tensor x_start,
    torch::Tensor noise,
    float sqrt_alpha_cumprod,
    float sqrt_one_minus_alpha_cumprod) {
    
    auto output = torch::empty_like(x_start);
    const int size = x_start.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    add_noise_3d_kernel<<<blocks, threads>>>(
        x_start.data_ptr<float>(),
        noise.data_ptr<float>(),
        output.data_ptr<float>(),
        sqrt_alpha_cumprod,
        sqrt_one_minus_alpha_cumprod,
        size
    );
    
    return output;
}

// Fused convolution + normalization + activation
__global__ void fused_conv_norm_silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // SiLU activation: x * sigmoid(x)
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
    }
}

torch::Tensor fused_silu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    fused_conv_norm_silu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}
"""

# C++ wrapper
cpp_source = """
torch::Tensor add_noise_3d_cuda(
    torch::Tensor x_start,
    torch::Tensor noise,
    float sqrt_alpha_cumprod,
    float sqrt_one_minus_alpha_cumprod);

torch::Tensor fused_silu_cuda(torch::Tensor input);
"""

class CUDAOps:
    """Custom CUDA operations wrapper"""
    def __init__(self):
        self.cuda_available = False
        self.module = None
        
        try:
            self.module = load_inline(
                name='cuda_ops',
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=['add_noise_3d_cuda', 'fused_silu_cuda'],
                verbose=False,
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                extra_cflags=['-O3']
            )
            self.cuda_available = True
            print("✓ Custom CUDA kernels compiled successfully")
        except Exception as e:
            print(f"⚠ CUDA compilation failed: {e}")
            print("  Falling back to PyTorch operations")
    
    def add_noise_3d(self, x_start, noise, sqrt_alpha, sqrt_one_minus):
        """Add noise using custom CUDA kernel or PyTorch fallback"""
        if self.cuda_available and x_start.is_cuda:
            return self.module.add_noise_3d_cuda(
                x_start.contiguous(),
                noise.contiguous(),
                float(sqrt_alpha),
                float(sqrt_one_minus)
            )
        else:
            # PyTorch fallback
            return sqrt_alpha * x_start + sqrt_one_minus * noise
    
    def fused_silu(self, x):
        """Fused SiLU activation"""
        if self.cuda_available and x.is_cuda:
            return self.module.fused_silu_cuda(x.contiguous())
        else:
            return torch.nn.functional.silu(x)


# Global CUDA ops instance
cuda_ops = CUDAOps()