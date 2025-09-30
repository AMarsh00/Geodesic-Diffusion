import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Path to your CelebA-HQ folder with images
INPUT_DIR = "/data5/accounts/marsh/.cache/kagglehub/datasets/badasstechie/celebahq-resized-256x256/versions/1/celeba_hq_256"
# Output directory for processed images
OUTPUT_DIR = "./celeba_hq_prepared"

# Desired resolution for training (adjust as needed)
TARGET_RESOLUTION = 64

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define image transform: resize to target resolution (you can add crop if needed)
transform = transforms.Compose([
    transforms.Resize((TARGET_RESOLUTION, TARGET_RESOLUTION)),
])

def prepare_dataset():
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"Found {len(image_files)} images to process.")

    for i, fname in enumerate(tqdm(image_files)):
        path = os.path.join(INPUT_DIR, fname)
        img = Image.open(path).convert("RGB")
        img = transform(img)
        save_path = os.path.join(OUTPUT_DIR, f"{i:06d}.png")
        img.save(save_path)

    print(f"Saved processed images to {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_dataset()
