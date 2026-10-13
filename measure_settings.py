import argparse
import csv
import json
import os
import cv2
import numpy as np

from enhance_dataset import enhance_image

#get folders and band width from command line
parser = argparse.ArgumentParser()
parser.add_argument("--images", required=True)
parser.add_argument("--ann", required=True)
parser.add_argument("--output", default="results")
parser.add_argument("--band", type=int, default=5)
args = parser.parse_args()

#settings to test, plus the original and global equalization to compare against
settings = [("original", None, None), ("global_eq", None, None)]
for c in [1, 2, 3, 4]:
    for t in [4, 8, 16]:
        settings.append((f"clip{c}_tile{t}", c, t))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.band + 1, 2 * args.band + 1))
rows = []
names = sorted(f for f in os.listdir(args.images) if f.lower().endswith(".jpg"))
for i, name in enumerate(names):
    img = cv2.imread(os.path.join(args.images, name), cv2.IMREAD_GRAYSCALE)
    ann = json.load(open(os.path.join(args.ann, name + ".json")))

    #some images have no teeth labeled so there is nothing to measure
    if len(ann["objects"]) == 0:
        print("no teeth labeled in", name)
        continue

    #draw every tooth polygon as its own mask
    teeth = []
    for obj in ann["objects"]:
        pts = np.array(obj["points"]["exterior"], dtype=np.int32)
        mask = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        teeth.append(mask)
    all_teeth = np.max(teeth, axis=0)

    #inner band is the tooth edge, outer band is the thin ring just outside it
    inner = []
    outer = []
    for mask in teeth:
        in_band = mask - cv2.erode(mask, kernel)
        out_band = cv2.dilate(mask, kernel) - mask
        #dont count pixels that belong to a neighbouring tooth
        out_band[all_teeth == 1] = 0
        if in_band.sum() > 0 and out_band.sum() > 0:
            inner.append(in_band == 1)
            outer.append(out_band == 1)
    tooth_pixels = all_teeth == 1

    #tooth cores stay 10 px away from every edge so they should look smooth
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    cores = cv2.erode(all_teeth, core_kernel) == 1

    for label, c, t in settings:
        if label == "original":
            out = img
        elif label == "global_eq":
            out = cv2.equalizeHist(img)
        else:
            out = enhance_image(img, c, t)

        #washout is the share of tooth pixels at full brightness
        washout = (out[tooth_pixels] == 255).mean() * 100
        #near white is a looser version, 250 and up
        near_white = (out[tooth_pixels] >= 250).mean() * 100

        #grain is how much fine detail is left after removing a blurred copy
        fine = out.astype(np.float32) - cv2.GaussianBlur(out, (0, 0), 2).astype(np.float32)
        grain = fine[cores].std()

        #edge contrast is tooth edge brightness minus the bone band outside it
        diffs = [out[a].mean() - out[b].mean() for a, b in zip(inner, outer)]
        rows.append([name, label, c, t, round(washout, 4), round(near_white, 4), round(float(np.mean(diffs)), 3), round(float(grain), 3), len(diffs)])

    if i % 50 == 0:
        print(f"{i}/{len(names)} images done")

#save one row per image per setting
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "per_image.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "setting", "clip", "tile", "washout_pct", "near_white_pct", "edge_contrast", "grain", "teeth"])
    w.writerows(rows)

#average each setting over all images
with open(os.path.join(args.output, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["setting", "clip", "tile", "washout_pct", "near_white_pct", "edge_contrast", "grain"])
    for label, c, t in settings:
        mine = [r for r in rows if r[1] == label]
        w.writerow([label, c, t, round(np.mean([r[4] for r in mine]), 4), round(np.mean([r[5] for r in mine]), 4), round(np.mean([r[6] for r in mine]), 3), round(np.mean([r[7] for r in mine]), 3)])
print("saved", args.output)
