import argparse
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#get image name and settings
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--clip", type=int, default=2)
parser.add_argument("--tile", type=int, default=8)
parser.add_argument("--raw", default="raw")
parser.add_argument("--enhanced", default="enhanced")
parser.add_argument("--out", default="comparisons")
args = parser.parse_args()

#load original and enhanced
folder = f"clip{args.clip}_tile{args.tile}"
orig = cv2.imread(os.path.join(args.raw, args.image), cv2.IMREAD_GRAYSCALE)
enh = cv2.imread(os.path.join(args.enhanced, folder, args.image), cv2.IMREAD_GRAYSCALE)

#plot side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
#vmin vmax so matplotlib doesnt rescale brightness
axes[0].imshow(orig, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("original")
axes[1].imshow(enh, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(f"clahe clip={args.clip} tile={args.tile}")
for ax in axes:
    ax.axis("off")
fig.suptitle(f"{args.image}  clipLimit={args.clip}  tileGridSize=({args.tile},{args.tile})")
plt.tight_layout()

#save figure
os.makedirs(args.out, exist_ok=True)
name = os.path.splitext(args.image)[0]
save_path = os.path.join(args.out, f"{name}_clip{args.clip}_tile{args.tile}.png")
plt.savefig(save_path, dpi=200)
print("saved", save_path)
