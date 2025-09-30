import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


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
        """
        Diffuse the data (adding noise) at timestep t
        x0: clean images
        t: tensor of shape [batch]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alphas_cumprod_t = self.alphas_cumprod.to(x0.device)[t]
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod_t).view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise


# ------------------------
# Sampling function
# ------------------------
@torch.no_grad()
def sample(model, scheduler, device, num_steps=1000, img_size=64, batch_size=4):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    for t in reversed(range(num_steps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        beta_t = scheduler.betas[t].to(device)
        alpha_t = scheduler.alphas[t].to(device)
        alpha_cumprod_t = scheduler.alphas_cumprod[t].to(device)

        pred_noise = model(x, t_batch)
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
        x = coef1 * (x - coef2 * pred_noise)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2  # [0,1]
    return x


# ------------------------
# Training loop
# ------------------------
def train():
    data_dir = "./celeba_hq_prepared"  # your dataset path here
    save_dir = "./vp_diffusion_outputs"
    os.makedirs(save_dir, exist_ok=True)

    image_size = 64
    batch_size = 32
    epochs = 2000
    lr = 2e-4
    num_timesteps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CelebAHQDataset(data_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetSD().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = VPScheduler(num_timesteps=num_timesteps)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x0 = batch.to(device)
            t = torch.randint(0, scheduler.num_timesteps, (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)
            xt = scheduler.q_sample(x0, t, noise)

            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"unet_epoch_{epoch+1}.pt"))

        # Save sample images
        samples = sample(model, scheduler, device, num_steps=num_timesteps, img_size=image_size, batch_size=4)
        save_image(samples, os.path.join(save_dir, f"samples_epoch_{epoch+1}.png"), nrow=2)


if __name__ == "__main__":
    train()
