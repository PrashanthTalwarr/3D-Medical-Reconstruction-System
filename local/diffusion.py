import torch
import torch.nn.functional as F
from local.cuda_ops import cuda_ops

class DDPM3D:
    """
    Denoising Diffusion Probabilistic Model for 3D Medical Images
    Based on: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    """
    def __init__(self, timesteps=200, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for diffusion process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance: q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        
        # Use custom CUDA kernel if available
        noisy_x = cuda_ops.add_noise_3d(
            x_start,
            noise,
            sqrt_alphas_cumprod_t,
            sqrt_one_minus_alphas_cumprod_t
        )
        
        return noisy_x, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute mean and variance of posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self.betas[t][:, None, None, None, None] * torch.sqrt(self.alphas_cumprod_prev[t][:, None, None, None, None]) * x_start +
            (1.0 - self.alphas_cumprod_prev[t][:, None, None, None, None]) * torch.sqrt(self.alphas[t][:, None, None, None, None]) * x_t
        ) / (1.0 - self.alphas_cumprod[t][:, None, None, None, None])
        
        posterior_variance = self.posterior_variance[t][:, None, None, None, None]
        
        return posterior_mean, posterior_variance
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)
        Sample x_{t-1} given x_t using the model
        """
        # Predict noise
        predicted_noise = model(x, t)
        
        # Predict x_0
        x_start = self.predict_start_from_noise(x, t, predicted_noise)
        x_start = torch.clamp(x_start, -1.0, 1.0)
        
        # Compute posterior mean and variance
        model_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x, t)
        
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, device=None):
        """
        Generate new samples from pure noise
        Full reverse diffusion process
        """
        if device is None:
            device = self.device
            
        batch_size = shape[0]
        
        # Start from pure noise (x_T ~ N(0, I))
        img = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
        
        return img
    
    @torch.no_grad()
    def sample_with_progress(self, model, shape, device=None):
        """Generate samples and return intermediate steps for visualization"""
        if device is None:
            device = self.device
            
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        
        imgs = []
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            
            # Save intermediate steps
            if i % (self.timesteps // 10) == 0:
                imgs.append(img.cpu())
        
        return img, imgs