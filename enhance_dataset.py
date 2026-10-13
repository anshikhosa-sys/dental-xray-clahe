import argparse
import os
import sys
import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def enhance_image(img, clip_limit=2.0, tile_grid_size=8):
    #color images get turned into grayscale first
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #apply clahe
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img)


def enhance_folder(input_dir, output_dir, clip_limit=2.0, tile_grid_size=8):
    in_abs = os.path.abspath(input_dir)
    out_abs = os.path.abspath(output_dir)

    #never write into or around the originals
    if out_abs == in_abs or out_abs.startswith(in_abs + os.sep) or in_abs.startswith(out_abs + os.sep):
        sys.exit("output folder can't be the input folder or overlap with it")

    count = 0
    for root, dirs, files in os.walk(in_abs):
        dirs.sort()
        for name in sorted(files):
            #skip non image files
            if not name.lower().endswith(IMAGE_EXTS):
                continue

            #read image as grayscale
            img = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("couldn't read", os.path.join(root, name))
                continue

            #same subfolder and filename on the output side
            rel = os.path.relpath(root, in_abs)
            save_dir = os.path.join(out_abs, rel)
            os.makedirs(save_dir, exist_ok=True)

            #save at max jpeg quality so training data loses as little as possible
            enhanced = enhance_image(img, clip_limit, tile_grid_size)
            cv2.imwrite(os.path.join(save_dir, name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 100])
            count += 1
    return count


if __name__ == "__main__":
    #get settings from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--tile", type=int, default=8)
    args = parser.parse_args()

    count = enhance_folder(args.input, args.output, args.clip, args.tile)

    #output
    print(f"enhanced {count} images with clip={args.clip} tile={args.tile} -> {args.output}")
