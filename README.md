# Dental X-ray CLAHE preprocessing

CLAHE contrast enhancement for panoramic dental X-rays, plus the scripts used to pick its settings.

Recommended settings: `clip_limit=2.0`, `tile_grid_size=8`.

A personal project exploring how to preprocess dental X-rays so tooth roots and jawbone edges are easier to see, for example before training a tooth segmentation model.

## Results

Measured over 595 labeled panoramic X-rays from the Kaggle dataset (3 of the 598 have no teeth labeled), using the tooth polygons as masks. See [Measuring settings](#measuring-settings) for how each number is computed.

| Setting | Washout at 255 (%) | Near white 250+ (%) | Edge contrast | Grain |
| --- | --- | --- | --- | --- |
| Original | 0.0071 | 0.02 | 11.5 | 2.55 |
| Global histogram equalization | 0.0103 | 0.73 | 17.8 | 3.30 |
| CLAHE clip 2, tile 4 | 0.0079 | 0.42 | 18.4 | 3.64 |
| **CLAHE clip 2, tile 8** | **0.0073** | **0.21** | **19.1** | **4.04** |
| CLAHE clip 3, tile 16 | 0.0074 | 0.15 | 21.1 | 5.26 |
| CLAHE clip 4, tile 4 | 0.0728 | 1.79 | 22.1 | 4.48 |

- Clip 2, tile 8 raises tooth edge contrast by 67% over the original while keeping washout at the original's level.
- Compared with global histogram equalization it has higher edge contrast in 468 of 595 images and the same or less near-white washout in 569. Global equalization does leave less grain (CLAHE is grainier in 566 of 595).
- Edge contrast keeps rising with the clip limit, but so does grain, so higher clips trade smoothness for edges. Clip 2, tile 8 has the highest edge contrast among settings with at most 0.25% near white and grain around 4. Those limits are a judgment call, not a hard optimum.
- The dataset's PNG masks are 4083 px wide and do not line up with the 2041 px X-rays, so the measurements use the JSON tooth polygons in `ann/` instead.

## Use it from Python

```python
import cv2
from enhance_dataset import enhance_image, enhance_folder

img = cv2.imread("xray.jpg", cv2.IMREAD_GRAYSCALE)
out = enhance_image(img, clip_limit=2.0, tile_grid_size=8)

# whole folder, keeps filenames, subfolders and image size
enhance_folder("raw/", "enhanced/", clip_limit=2.0, tile_grid_size=8)
```

`enhance_image` takes a grayscale or BGR image and returns an 8-bit grayscale image of the same height and width. It never changes the array you pass in.

## Use it from the command line

```
python enhance_dataset.py --input raw/ --output enhanced/ --clip 2 --tile 8
```

- Walks the input folder recursively and writes each image to the same relative path and filename under the output folder.
- Skips anything that is not .jpg, .jpeg or .png (for example .DS_Store).
- Refuses to run if the output folder is the input folder or overlaps with it, so originals are never overwritten.
- JPEGs are saved at quality 100.

## Tests

```
pip install -r requirements.txt
python -m pytest
```

The tests check that output keeps filenames, folder structure and image size, that non-images are skipped, that originals are left byte-for-byte unchanged, and that an overlapping output folder is refused.

## Measuring settings

`measure_settings.py` scores a grid of settings (clipLimit 1-4 x tileGridSize 4, 8, 16), the original, and global histogram equalization using the tooth polygons in the dataset's `ann/` JSON files:

- washout: % of tooth pixels at 255 (and at 250 or more)
- edge contrast: mean brightness of a 5 px band just inside each tooth edge minus a 5 px band just outside it (other teeth excluded)
- grain: std of fine detail (image minus a Gaussian blur, sigma 2) inside tooth cores at least 10 px from any edge

```
python measure_settings.py --images "<dataset>/d2/img" --ann "<dataset>/d2/ann" --output results
python plot_settings.py
```

`compare.py` saves an original vs enhanced figure for one image from a sweep folder.

Data: [Kaggle Teeth Segmentation on dental X-ray images](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images). The dataset and any images are not included in this repo.
