import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# ------------------------
# Dataset
# ------------------------
class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

# ------------------------
# Time embedding helper
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

# ------------------------
# Residual Block
# ------------------------
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

# ------------------------
# Downsample and Upsample blocks
# ------------------------
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, 2, 1)  # stride 2, kernel 4

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, 2, 1)  # stride 2, kernel 4

    def forward(self, x):
        return self.conv(x)

# ------------------------
# UNetSD Model
# ------------------------
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

        # Down blocks
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

        # Middle blocks
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Up blocks
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

        # Down path
        residuals = []
        for block, down in zip(self.downs, self.downsamples):
            x = block(x, t_emb)
            residuals.append(x)
            x = down(x)

        # Middle
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # Up path
        for upsample, block in zip(self.upsamples, self.ups):
            x = upsample(x)
            res = residuals.pop()
            x = torch.cat([x, res], dim=1)  # concat skip connection
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
        alphas_cumprod_t = self.alphas_cumprod.to(x0.device)[t]
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod_t).view(-1,1,1,1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod_t).view(-1,1,1,1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

# ------------------------
# Reverse diffusion step (better sampling)
# ------------------------
@torch.no_grad()
def p_sample(model, scheduler, x_t, t):
    device = x_t.device

    betas_t = scheduler.betas.to(device)[t]
    alphas_t = scheduler.alphas.to(device)[t]
    alphas_cumprod_t = scheduler.alphas_cumprod.to(device)[t]

    pred_noise = model(x_t, t)
    coef1 = 1 / torch.sqrt(alphas_t)
    coef2 = betas_t / torch.sqrt(1 - alphas_cumprod_t)
    mean = coef1[:, None, None, None] * (x_t - coef2[:, None, None, None] * pred_noise)

    noise = torch.randn_like(x_t) if (t > 0).any() else torch.zeros_like(x_t)
    sigma = torch.sqrt(betas_t)

    sample = mean + sigma[:, None, None, None] * noise
    return sample

# ------------------------
# Build approximate k-NN graph with FAISS
# ------------------------
def build_approx_knn_graph(images, k=5):
    N = images.shape[0]
    flat_imgs = images.view(N, -1).cpu().numpy().astype(np.float32)
    dim = flat_imgs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(flat_imgs)
    distances, neighbors = index.search(flat_imgs, k+1)

    rows, cols, data = [], [], []
    for i in range(N):
        for j in range(1, k+1):
            rows.append(i)
            cols.append(neighbors[i, j])
            data.append(distances[i, j])
    graph = csr_matrix((data, (rows, cols)), shape=(N, N))
    return graph

def find_geodesic_midpoint(dataset, idx_start, idx_end, k=5):
    print("Building k-NN graph for geodesic using FAISS ANN...")
    images = torch.stack([dataset[i] for i in range(len(dataset))])
    graph = build_approx_knn_graph(images, k=k)
    print("Running shortest path Dijkstra...")
    dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=idx_start, return_predecessors=True)
    path = []
    cur = idx_end
    while cur != -9999:
        path.append(cur)
        cur = predecessors[cur]
    path = path[::-1]
    if len(path) == 0:
        print("No path found, returning midpoint by simple average")
        return (dataset[idx_start] + dataset[idx_end]) / 2
    midpoint_idx = path[len(path)//2]
    print(f"Geodesic midpoint index: {midpoint_idx}")
    return dataset[midpoint_idx]

# ------------------------
# Forward diffusion integral helper (for animation)
# ------------------------
def forward_diffuse_integral(x0, t, betas):
    device = x0.device
    beta_cumsum = torch.cumsum(betas, dim=0)
    integral_beta = beta_cumsum[t]
    exp_term = torch.exp(-0.5 * integral_beta)
    noise = torch.randn_like(x0)
    return exp_term[:, None, None, None] * x0 + torch.sqrt(1 - exp_term**2)[:, None, None, None] * noise

# ------------------------
# Animation helpers
# ------------------------
def show_img_grid(tensors, title=""):
    grid_img = make_grid(tensors, nrow=int(math.sqrt(len(tensors))))
    np_img = grid_img.permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 0.5 + 0.5).clip(0, 1)  # denormalize for display
    plt.imshow(np_img)
    plt.title(title)
    plt.axis("off")

# ------------------------
# Animation functions
# ------------------------
def animate_forward_diffusion(dataset, scheduler):
    fig, ax = plt.subplots(figsize=(8, 8))
    indices = np.random.choice(len(dataset), 16, replace=False)
    x0 = torch.stack([dataset[i] for i in indices]).to(device)
    betas = scheduler.betas.to(device)
    num_frames = 50

    def update(frame):
        ax.clear()
        t = torch.tensor([int(frame * scheduler.num_timesteps / num_frames)] * len(x0), device=device)
        noisy_imgs = scheduler.q_sample(x0, t)
        show_img_grid(noisy_imgs, f"Forward Diffusion Step {frame}")
        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100)
    anim.save("forward_diffusion.gif", writer=PillowWriter(fps=10))
    plt.close()
    print("Forward diffusion animation saved as forward_diffusion.gif")

def animate_reverse_diffusion(model, scheduler, dataset):
    fig, ax = plt.subplots(figsize=(8, 8))
    indices = np.random.choice(len(dataset), 4, replace=False)
    x_start = torch.stack([dataset[i] for i in indices]).to(device)
    x_end = torch.stack([dataset[(i+10) % len(dataset)] for i in indices]).to(device)

    # Get geodesic midpoint image in latent space
    midpoint_img = find_geodesic_midpoint(dataset, indices[0], (indices[0] + 10) % len(dataset))

    num_frames = 1000
    model.eval()

    # Pre-generate noise tensor for initial sample
    x_t = torch.randn_like(midpoint_img.unsqueeze(0)).repeat(len(indices), 1, 1, 1).to(device)

    def update(frame):
        ax.clear()
        t_val = scheduler.num_timesteps - 1 - int(frame * scheduler.num_timesteps / num_frames)
        t = torch.tensor([t_val] * len(indices), device=device)
        nonlocal x_t
        x_t = p_sample(model, scheduler, x_t, t)
        show_img_grid(x_t, f"Reverse Diffusion Step {frame}")
        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=False)
    anim.save("reverse_diffusion.gif", writer=PillowWriter(fps=10))
    plt.close()
    print("Reverse diffusion animation saved as reverse_diffusion.gif")
def animate_reverse_diffusion_multi(model, scheduler, dataset):
    indices = np.random.choice(len(dataset), 4, replace=False)
    x_start = torch.stack([dataset[i] for i in indices]).to(device)

    # Labels for the four images - you can customize these however you want
    image_titles = ["Endpoint 1", "Linear interp", "Geodesic midpoint", "Endpoint 2"]

    # For demonstration, compute linear interp and geodesic midpoint images for the middle two
    # You can replace these with actual interpolations or midpoints as needed
    # Here: just placeholders (copying endpoints)
    # If you want actual linear/geodesic midpoints, compute them accordingly
    x_linear = (x_start[0] + x_start[3]) / 2
    midpoint_img = find_geodesic_midpoint(dataset, indices[0], indices[3])
    x_geo = midpoint_img

    x_linear = x_linear.to(device)
    x_geo = x_geo.to(device)
    x_imgs = torch.stack([x_start[0], x_linear, x_geo, x_start[3]])
    #x_imgs = torch.stack([x_start[0], x_linear, x_geo, x_start[3]]).to(device)

    timestep_lengths = [999, 500, 250, 100, 50, 0]

    for steps in timestep_lengths:
        if steps == 0:
            # Just show clean images with titles above each image
            fig, ax = plt.subplots(figsize=(8, 8))
            grid_img = make_grid(x_imgs, nrow=2, padding=10)
            np_img = grid_img.permute(1, 2, 0).cpu().numpy()
            np_img = (np_img * 0.5 + 0.5).clip(0, 1)
            ax.imshow(np_img)
            ax.axis("off")

            # Add titles above each image
            W = np_img.shape[1] // 2
            H = np_img.shape[0] // 2
            positions = [(W//2, 20), (3*W//2, 20), (W//2, H + 20), (3*W//2, H + 20)]
            for pos, title in zip(positions, image_titles):
                ax.text(pos[0], pos[1], title, color='white', fontsize=14, fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
            plt.tight_layout()
            plt.savefig(f"reverse_diffusion_t{steps}.png")
            plt.close()
            print(f"Saved clean images as reverse_diffusion_t{steps}.png")
            continue

        model.eval()
        with torch.no_grad():
            # Forward noise to timestep 'steps' for each image separately
            t_forward = torch.tensor([steps] * len(x_imgs), device=device)
            x_t = scheduler.q_sample(x_imgs, t_forward)

        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            ax.clear()
            t_val = steps - frame
            t_val = max(t_val, 0)
            t = torch.tensor([t_val] * len(x_imgs), device=device)

            nonlocal x_t
            if t_val > 0:
                x_t = p_sample(model, scheduler, x_t, t)
            else:
                x_t = x_imgs

            grid_img = make_grid(x_t, nrow=2, padding=10)
            np_img = grid_img.permute(1, 2, 0).cpu().numpy()
            np_img = (np_img * 0.5 + 0.5).clip(0, 1)
            ax.imshow(np_img)
            ax.axis("off")

            # Add titles above each image
            W = np_img.shape[1] // 2
            H = np_img.shape[0] // 2
            positions = [(W//2, 20), (3*W//2, 20), (W//2, H + 20), (3*W//2, H + 20)]
            for pos, title in zip(positions, image_titles):
                ax.text(pos[0], pos[1], title, color='white', fontsize=14, fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

            ax.set_title(f"Reverse Diffusion: timestep = {t_val}", fontsize=16, pad=30)
            plt.tight_layout()

        anim = FuncAnimation(fig, update, frames=steps + 1, interval=100, repeat=False)
        anim.save(f"reverse_diffusion_t{steps}.gif", writer=PillowWriter(fps=10))
        plt.close()
        print(f"Saved reverse diffusion animation starting at timestep {steps} as reverse_diffusion_t{steps}.gif")


# ------------------------
# Main entry
# ------------------------
def main():
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CelebAHQDataset("celeba_hq_prepared/", image_size=64)

    model = UNetSD().to(device)
    scheduler = VPScheduler(num_timesteps=1000)

    checkpoint_path = "vp_diffusion_outputs/unet_epoch_1000.pt"
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, exiting.")
        return

    # Animate forward diffusion
    animate_forward_diffusion(dataset, scheduler)

    # Animate reverse diffusion using geodesic midpoint
    animate_reverse_diffusion_multi(model, scheduler, dataset)

if __name__ == "__main__":
    main()
