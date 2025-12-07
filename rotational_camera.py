import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import os
import pickle
from tqdm import tqdm

def color_from_index(i, total=20):
    cmap = plt.get_cmap("tab20")   # or tab10, Set3, etc.
    return cmap(i % cmap.N) 

def find_3d_point(mask: np.ndarray, map_file, visualize=False, output_path="output_with_categories.las"):
    with open(map_file,"rb") as f:
        data = pickle.load(f)

    colors = []
    xyz = []
    cats = []

    size_y, size_x = mask.shape
    for x in tqdm(range(size_y)):
        for y in range(size_x):
            # if y % 4 != 1 or x % 4 != 1:
            #     continue
            point3D = data["interpolated_points"][y, x]
            xyz.append(point3D)
            colors.append(color_from_index(mask[y, x]))

            cat = mask[y, x]
            cats.append(cat)


    np.save("colors.npy", colors)
    np.save("xyz.npy", xyz)
    np.save("cats.npy", cats)

    header = laspy.LasHeader(point_format=8, version="1.4")
    las = laspy.LasData(header)

    xyz = np.array(xyz)
    colors = np.array(colors)
    cats = np.array(cats)

    # Assign XYZ
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # Assign classification
    las.classification = cats

    # Assign colors (R,G,B)
    las.red = colors[:, 0]
    las.green = colors[:, 1]
    las.blue = colors[:, 2]

    # Save file
    las.write(output_path)
    return output_path


def generate_images(path, img_size=4048, blurring=False, sigma=1, n_cameras=16, camera_height=200000):
    # ------------------------------
    # 1. Load LAS
    # ------------------------------
    las = laspy.read(path)
    points = np.vstack((las.X, las.Y, las.Z)).T
    colors = np.vstack((las.red, las.green, las.blue)).T / 65535

    # ------------------------------
    # 2. Define circular camera positions
    # ------------------------------
    center = points.mean(axis=0)       # rotation around point cloud center
    radius = np.linalg.norm(np.ptp(points, axis=0)) * 1.5
    height = center[2] + camera_height            # constant height above ground

    angles = np.linspace(0, 2*np.pi, n_cameras, endpoint=False)
    camera_positions = np.array([
        [center[0] + radius*np.cos(a), center[1] + radius*np.sin(a), height] for a in angles
    ])

    # ------------------------------
    # 3. Output folders
    # ------------------------------
    image_folder = "circular_projections/images"
    map_folder = "circular_projections/pixel_maps"
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(map_folder, exist_ok=True)

    image_files = []
    map_files = []

    # ------------------------------
    # 4. Render projections with smooth interpolation
    # ------------------------------
    for i, cam_pos in enumerate(tqdm(camera_positions)):
        # Camera axes
        forward = center - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        # Transform points to camera coordinates
        relative = points - cam_pos
        x_cam = np.dot(relative, right)
        y_cam = np.dot(relative, up)
        z_cam = np.dot(relative, forward)

        # Keep points in front of camera
        mask = z_cam > 0
        x_cam = x_cam[mask]
        y_cam = y_cam[mask]
        points_cam = points[mask]
        colors_cam = colors[mask]
        z_cam = z_cam[mask]

        # Normalize to [0,1] for pixel coordinates
        x_norm = (x_cam - x_cam.min()) / (x_cam.max() - x_cam.min())
        y_norm = (y_cam - y_cam.min()) / (y_cam.max() - y_cam.min())
        x_pix = (x_norm * (img_size-1)).astype(int)
        y_pix = (y_norm * (img_size-1)).astype(int)

        # ------------------------------
        # Gaussian splatting
        img = np.zeros((img_size, img_size, 3))
        img[:, :, 0] = 0.67
        img[:, :, 1] = 0.87
        img[:, :, 2] = 0.9
        depth_map = np.full((img_size, img_size), np.inf)
        pixel_map = -np.ones((img_size, img_size), dtype=int)

        for idx in range(len(points_cam)):
            px = x_pix[idx]
            py = img_size - 1 - y_pix[idx]
            if z_cam[idx] < depth_map[py, px]:
                depth_map[py, px] = z_cam[idx]
                pixel_map[py, px] = idx
                img[py, px, :] = colors_cam[idx]

        if blurring:
            for c in range(3):
                img[:,:,c] = gaussian_filter(img[:,:,c], sigma=sigma)

        # ------------------------------
        # Continuous 3D map
        grid_y, grid_x = np.mgrid[0:img_size, 0:img_size]
        grid_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        valid = pixel_map >= 0
        valid_coords = np.stack(np.nonzero(valid), axis=-1)
        valid_points = points_cam[pixel_map[valid]]

        tree = cKDTree(valid_coords)
        _, idxs = tree.query(grid_coords, k=1)
        interpolated_points = valid_points[idxs].reshape(img_size, img_size, 3)

        # ------------------------------
        # Save image and mapping
        output_image = os.path.join(image_folder, f"proj_{i:03d}.png")
        image_files.append(output_image)
        plt.imsave(output_image, img)
        output_meta = os.path.join(map_folder, f"pixel_map_{i:03d}.pkl")
        map_files.append(output_meta)
        with open(output_meta, "wb") as f:
            pickle.dump({
                "pixel_map": pixel_map,
                "points_cam": points_cam,
                "colors_cam": colors_cam,
                "depth_map": depth_map,
                "interpolated_points": interpolated_points
            }, f)
    return image_files, map_files


if __name__ == '__main__':
    # img_folder, map_folder = generate_images("Chmura100.las", n_cameras=1)
    # print(f"Saved to {img_folder} and {map_folder}")
    mask = np.load("./segments.txt.npz")["arr_0"]
    find_3d_point(mask, "circular_projections/pixel_maps/pixel_map_000.pkl")
    