import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ------------------------
# UNetSD Components
# ------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        device = timestep.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timestep[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_skip_conv=False):
        super().__init__()
        self.use_skip_conv = use_skip_conv
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if in_channels != out_channels or use_skip_conv:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.relu(h)
        h = self.conv1(h)
        time_emb = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.norm2(h)
        h = self.relu(h)
        h = self.conv2(h)
        return h + self.skip_conv(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class UNetSD(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([
            ResidualBlock(base_channels, base_channels, time_emb_dim),
            ResidualBlock(base_channels, base_channels * 2, time_emb_dim, use_skip_conv=True),
            ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, use_skip_conv=True),
        ])
        self.downsamples = nn.ModuleList([
            Downsample(base_channels),
            Downsample(base_channels * 2),
            Downsample(base_channels * 4),
        ])
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.upsamples = nn.ModuleList([
            Upsample(base_channels * 4),
            Upsample(base_channels * 2),
            Upsample(base_channels),
        ])
        self.ups = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim, use_skip_conv=True),
            ResidualBlock(base_channels * 4, base_channels, time_emb_dim, use_skip_conv=True),
            ResidualBlock(base_channels * 2, base_channels, time_emb_dim, use_skip_conv=True),
        ])
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_relu = nn.ReLU()
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        residuals = []
        for block, down in zip(self.downs, self.downsamples):
            x = block(x, t_emb)
            residuals.append(x)
            x = down(x)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        for upsample, block in zip(self.upsamples, self.ups):
            x = upsample(x)
            res = residuals.pop()
            x = torch.cat([x, res], dim=1)
            x = block(x, t_emb)
        x = self.out_norm(x)
        x = self.out_relu(x)
        x = self.out_conv(x)
        return x

# ------------------------
# VP Scheduler
# ------------------------
class VPScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
    
        alphas_cumprod = self.alphas_cumprod.to(x0.device)
    
        if isinstance(t, int):
            t = torch.tensor([t], device=x0.device)
    
        # Handle single or batched t
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)
    
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

# ------------------------
# DDIM deterministic sampling
# ------------------------
@torch.no_grad()
def ddim_sample(model, scheduler, x_T, timesteps, eta=0.0):
    x_t = x_T
    device = x_t.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t_prev]
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        t_tensor = torch.tensor([t], device=device)
        epsilon_theta = model(x_t, t_tensor)
        
        x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t
        
        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        sigma_t_val = sigma_t.item()
        
        if sigma_t_val > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        x_t = sqrt_alpha_t_prev * x0_pred + torch.sqrt(1 - alpha_t_prev - sigma_t_val**2) * epsilon_theta + sigma_t * noise
    
    return x_t

# ------------------------
# Load and transform image
# ------------------------
def load_image(path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

# ------------------------
# Show images helper
# ------------------------
def show_images(original, noised, denoised):
    def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    imgs = [original, noised, denoised]
    titles = ["Original", "Noised", "Denoised"]
    plt.figure(figsize=(12, 4))
    for i, img in enumerate(imgs):
        plt.subplot(1, 3, i + 1)
        plt.imshow(denorm(img))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ------------------------
# Main function
# ------------------------
def inspect_ddim_consistency(x0, x_t, t, model, scheduler):
    device = x0.device
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    alpha_t = alphas_cumprod[t]
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

    # Predict noise from the model
    with torch.no_grad():
        epsilon_theta = model(x_t, torch.tensor([t], device=device))

    # Ground truth noise used in forward pass
    true_epsilon = (x_t - sqrt_alpha_t * x0) / sqrt_one_minus_alpha_t

    # Reconstruct x0 from predicted noise
    x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_theta) / sqrt_alpha_t

    mse_noise = torch.mean((true_epsilon - epsilon_theta) ** 2).item()
    mse_x0 = torch.mean((x0 - x0_pred) ** 2).item()

    print(f"MSE between model epsilon and true epsilon: {mse_noise:.6f}")
    print(f"MSE between predicted x0 and true x0: {mse_x0:.6f}")

def interpolate_noisy_images(A, B, theta_vals):
    images = []
    for theta in theta_vals:
        x_theta = A * np.cos(theta * np.pi/2) + B * np.sin(theta * np.pi/2)
        images.append(x_theta)
    return images
"""def interpolate_noisy_images(A, B, theta_vals):
    images = []
    for theta in theta_vals:
        x_theta = A * theta + B * (1-theta)
        images.append(x_theta)
    return images"""

def run_reverse_diffusion_animation(model, scheduler, noisy_images, timesteps, device):
    """
    Runs reverse diffusion on a list of noisy images and collects frames.
    """
    frames = []
    for i, x_T in enumerate(noisy_images):
        print(f"Reverse diffusion for frame {i+1}/{len(noisy_images)}")
        x_denoised = ddim_sample(model, scheduler, x_T.to(device), timesteps)
        frames.append(x_denoised.cpu().squeeze(0))
    return frames

def animate_frames(frames):
    """
    Creates an animation of generated frames using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    
    im = ax.imshow(denorm(frames[0]))
    ax.axis('off')

    def update(i):
        im.set_array(denorm(frames[i]))
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)
    #plt.show()
    anim.save("interpolation.gif")
    return anim
    
def animate_multiple_sequences(all_frames_list, n_cols=2):
    """
    Create a grid of subplots for multiple frame sequences that update in sync.
    Each entry in all_frames_list is a list of frames for one interpolation.
    """
    n_sequences = len(all_frames_list)
    n_frames = len(all_frames_list[0])
    n_rows = (n_sequences + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case of 1 row

    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    ims = []
    for ax, frames in zip(axes, all_frames_list):
        im = ax.imshow(denorm(frames[0]))
        ax.axis('off')
        ims.append(im)

    def update(frame_idx):
        for im, frames in zip(ims, all_frames_list):
            im.set_array(denorm(frames[frame_idx]))
        return ims

    anim = FuncAnimation(fig, update, frames=n_frames, interval=150)
    plt.tight_layout()
    anim.save("multi_interpolation.gif")
    plt.close(fig)
    print("Saved animation as multi_interpolation.gif")
    return anim
    
def find_orthogonal_vector_to_tangent(V_t):
    """
    Find a vector orthogonal to each of the vectors in V_t using projection.
    
    Args:
        V_t (torch.Tensor): (Q, d) tensor where Q is the number of tangent vectors and d is the vector dimension.
    
    Returns:
        V_n (torch.Tensor): (Q, d) tensor where each vector is orthogonal to the corresponding V_t vector.
    """
    Q, d = V_t.shape
    V_n = torch.zeros_like(V_t)  # Initialize normal space vectors

    for i in range(Q):
        tangent_vector = V_t[i]
        
        # Create a random vector in the same space
        random_vector = torch.randn_like(tangent_vector)
        
        # Project random_vector onto tangent_vector to get the component along V_t
        projection = torch.dot(random_vector, tangent_vector) / torch.dot(tangent_vector, tangent_vector) * tangent_vector
        
        # Subtract the projection from random_vector to get the orthogonal component
        V_n[i] = random_vector - projection
    
    return V_n
    
def estimate_tangent_normal_space(x, model, scheduler, timesteps, Q=32, epsilon=0.05):
    """
    Estimate tangent and normal space at point x (image) using directional derivatives.
    
    Args:
        x (torch.Tensor): shape (1, C, H, W)
        Q (int): number of random directions
        epsilon (float): small perturbation parameter

    Returns:
        V_t (torch.Tensor): Tangent space basis vectors (Q x C x H x W)
        V_n (torch.Tensor): Normal space basis vectors ((d - Q) x C x H x W)
        S (torch.Tensor): Singular values
    """
    B, C, H, W = x.shape
    d = C * H * W
    theta = epsilon * np.pi / 2
    sin_theta = np.sin(theta)
    device = x.device
    
    # Collect directional derivatives
    R = []
    for _ in range(Q):
        N = torch.randn_like(x)
        D_N = (x * np.cos(theta) + N * sin_theta - x) / epsilon
        D_N = ddim_sample(model, scheduler, D_N.to(device), timesteps)
        R.append(D_N.flatten())
    
    R = torch.stack(R, dim=0)  # (Q, d)
    
    # SVD: R = U S V^T
    # V: right singular vectors (d x d)
    U, S, Vh = torch.linalg.svd(R.T, full_matrices=False)
    
    # Top Q rows of V^T form the tangent space
    V_t = Vh[:Q].reshape(Q, C, H, W)
    
     # Flatten V_t for computation
    V_t_flattened = V_t.flatten(1)
    
    # Find normal vectors that are orthogonal to V_t
    V_n_flattened = find_orthogonal_vector_to_tangent(V_t_flattened)  # (Q, d)
    
    # Reshape back to image space
    V_n = V_n_flattened.reshape(Q, C, H, W)
    
    return V_t, V_n, S
    
def visualize_directions(directions, title_prefix="Direction", n=6):
    n = min(n, len(directions))
    plt.figure(figsize=(12, 2))
    for i in range(n):
        vec = directions[i]
        img = (vec * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"{title_prefix} {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
def plot_singular_values_vs_epsilon(epsilons, singular_values_list):
    """
    Plots the top 5 singular values versus epsilon values.

    Args:
        epsilons (list): List of epsilon values.
        singular_values_list (list): List of singular values corresponding to each epsilon.
    """
    # Extract top 5 singular values for each epsilon
    top_singular_values = np.array([S.cpu().numpy() for S in singular_values_list])  # Shape: (N, 5)

    # Plot each of the top 5 singular values
    plt.figure(figsize=(8, 6))
    for i in range(5):
        plt.plot(epsilons, top_singular_values[:, i], marker='o', linestyle='-', label=f'Top {i+1} Singular Value')

    # Customizing the plot
    plt.title('Singular Values vs Epsilon')
    plt.xlabel('Epsilon')
    plt.ylabel('Singular Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path_A = "celeba_hq_prepared/000000.png"  # Image A
    image_path_B = "celeba_hq_prepared/000001.png"  # Image B

    x0_A = load_image(image_path_A).to(device)
    x0_B = load_image(image_path_B).to(device)
    print(f"Loaded images shapes: {x0_A.shape}, {x0_B.shape}")

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = "vp_diffusion_outputs/unet_epoch_1000.pt"
    if os.path.isfile(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print("Loaded trained model.")
    else:
        print("Checkpoint not found. Using randomly initialized model.")
    model.eval()
    
    T = 999
    t = torch.tensor([T], device=device)
    n_frames = 1
    theta_vals = np.linspace(0, 0.01, n_frames)
    timesteps = list(range(T, -1, -1))

    x = load_image("celeba_hq_prepared/000000.png").to(device)  # shape (1, 3, 64, 64)
    x = torch.randn_like(x)
    """V_t, V_n, S = estimate_tangent_normal_space(x, model, scheduler, timesteps, Q=32, epsilon=0.05)
    
    print(f"Tangent space basis shape: {V_t.shape}")  # (32, 3, 64, 64)
    print(f"Normal space basis shape: {V_n.shape}")   # (~12256, 3, 64, 64)
    print("Top singular values:", S[:5])
    
    visualize_directions(V_t, title_prefix="Tangent")
    visualize_directions(V_n, title_prefix="Normal")
    
    # Select a random normal vector and animate
    random_idx = np.random.randint(0, len(V_n))  # Pick a random normal vector from V_n
    normal_vector = V_n[random_idx].unsqueeze(0).to(device)  # Make it a batch of size 1

    # Animate movement along the normal vector
    n_frames = 60  # Number of frames in the animation
    alpha_values = np.linspace(0, 10, n_frames)  # Interpolate along the normal vector
    frames = []

    y = ddim_sample(model, scheduler, x, timesteps)
    for alpha in alpha_values:
        moved_image = y + alpha * normal_vector  # Move along the normal direction
        moved_image = moved_image.clamp(-1, 1)  # Ensure the pixel values are within valid range
        frames.append(moved_image.cpu().squeeze(0))

    # Create and save animation
    anim = animate_frames(frames)"""
    
    epsilon_values = np.linspace(0.0001, 0.01, 10)  # Epsilon range from 0.01 to 0.1
    singular_values_list = []  # To store top singular values for each epsilon

    for epsilon in epsilon_values:
        print(f"Running experiment with epsilon = {epsilon}")
        # Estimate tangent and normal space with different epsilon values
        V_t, V_n, S = estimate_tangent_normal_space(x, model, scheduler, timesteps, Q=32, epsilon=epsilon)
        singular_values_list.append(S)  # Store the singular values

    # Plot the top singular values vs epsilon
    plot_singular_values_vs_epsilon(epsilon_values, singular_values_list)

    """N = 16
    
    # Step 1: Base noise
    base_noise = torch.randn_like(x0_A)
    
    # Step 2: Generate N other noises
    noise_list = [torch.randn_like(x0_A) for _ in range(N)]
    
    # Step 3: For each additional noise, interpolate with base_noise, reverse diffuse, collect frames
    all_interpolated_frames = []
    
    for i, noise in enumerate(noise_list):
        print(f"Interpolating noise {i+1}/{N}")
        
        x_noised_A = scheduler.q_sample(x0_A, t, base_noise)
        x_noised_B = scheduler.q_sample(x0_B, t, noise)
        
        noisy_interp_images = interpolate_noisy_images(x_noised_A.cpu(), x_noised_B.cpu(), theta_vals)
        frames = run_reverse_diffusion_animation(model, scheduler, noisy_interp_images, small_timesteps, device)
        all_interpolated_frames.append(frames)
    
    # Step 4: Animate all frame sequences as subplots
    animate_multiple_sequences(all_interpolated_frames, n_cols=2)"""

if __name__ == "__main__":
    main()
