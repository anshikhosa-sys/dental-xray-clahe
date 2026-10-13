# Dental X-ray CLAHE preprocessing

CLAHE contrast enhancement for panoramic dental X-rays, plus the scripts used to pick its settings.

Recommended settings: `clip_limit=2.0`, `tile_grid_size=8`.

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
