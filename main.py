

import argparse
from tqdm import tqdm
from image_segmentation_2d import Segmentator, segment_image
from rotational_camera import generate_images, find_3d_point
import numpy as np
import os


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('filename')



def main(filename):
    segmentator = Segmentator()
    print("Model found")
    
    image_paths, map_paths = generate_images(filename, n_cameras=2)
    print(image_paths)

    masks_dir = "masks"
    os.makedirs(masks_dir, exist_ok=True)
    for i, (image, map_file) in tqdm(enumerate(zip(image_paths, map_paths))):
        label_mask = segment_image(segmentator, image)
        if i == 0:
            np.savez_compressed(f"{masks_dir}/mask{i}", label_mask)
        # this for now is interactive
        return find_3d_point(mask=label_mask, map_file=map_file)