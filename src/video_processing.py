# In[0]:
# Import the libraries
import os
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from matplotlib import patches as patches

from cp_hw6 import pixel2ray, set_axes_equal

# In[0]:
# Define the methods


def get_x(a, b, c, y):
    # ax + by + c = 0
    return int(-(b * y + c) / a)


def get_y(a, b, c, x):
    # ax + by + c = 0
    return int(-(a * x + c) / b)


def line_function_to_end_points(a, b, c, y_start, y_end):
    # ax + by + c = 0
    x_start = get_x(a, b, c, y_start)
    x_end = get_x(a, b, c, y_end)
    vx = [x_start, x_end]
    vy = [y_start, y_end]

    return vx, vy


def find_zero_crossings(frame, x_start, x_end, y_start, y_end):
    clip = frame[x_start:x_end, y_start:y_end]
    zero_crossings = np.where(np.diff(np.sign(clip), axis=1) >= 2)

    xs, ys = zero_crossings
    xs += x_start
    ys += y_start

    locs = np.concatenate((ys.reshape(-1, 1), xs.reshape(-1, 1),
                          np.ones_like(xs).reshape(-1, 1)), axis=1)

    return locs


def compute_intersection(l0, l, p0, n):
    # l0: [3, 1] vector: camera position under the plane coordinate
    # l: [3, 1] vector: ray direction
    # p0: a point on the plane
    # n: plane normal
    # return: p: intersection under the plane coordinate

    t = np.dot((p0 - l0), n) / np.dot(l, n)
    p = l0 + l * t

    return p


def cam2world(point, R, t):
    point_copy = point.reshape(-1)
    t_copy = t.reshape(-1)
    return R.T @ (point_copy - t_copy)


def world2cam(point, R, t):
    point_copy = point.reshape(-1)
    t_copy = t.reshape(-1)
    return R @ point_copy + t_copy


def camera2plane(
        frame_idx,
        shadow_edge,
        y1, y2,
        mtx, dist, R, t,
        cam, p0, n):

    a, b, c = shadow_edge

    # Pick two points on the horizontal line
    x1, x2 = get_x(a, b, c, y1), get_x(a, b, c, y2)

    points = np.array([
        [x1, y1], [x2, y2]
    ]).astype(np.float32)

    # Convert the 2D image points to rays
    rays = pixel2ray(points, mtx, dist)  # [2 x 1 x 3]
    rays = rays.transpose(0, 2, 1)  # [2 x 3 x 1]

    # Convert from camera coordinate to the plane coordinate
    rays = (R.T @ rays).reshape(2, 3)

    # Compute the intersection of the rays and the plane
    inter1 = compute_intersection(cam, rays[0], p0, n)
    inter2 = compute_intersection(cam, rays[1], p0, n)

    return inter1, inter2


# In[1]:
# Load the object images

# data_dir = '../data/slipper'
# data_dir = '../data/frog'
data_dir = '../data/bottle'

imgs = []
color_imgs = []
img_files = []
for img_file in os.listdir(data_dir):
    img_files.append(img_file)
img_files.sort()

for img_file in img_files:
    img = (cv2.imread(os.path.join(data_dir, img_file)) / 255.0).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color_imgs.append(img)
    imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

imgs = np.array(imgs)
color_imgs = np.array(color_imgs)

num_imgs = imgs.shape[0]
h, w = imgs[0].shape

# In[2]:
# Create the difference images

# Create the maximum intensity image and the minimum intensity image
I_max = imgs.max(axis=0)
I_min = imgs.min(axis=0)

# Create the shadow threshold image
I_shadow = (I_max + I_min) / 2

# Create the difference images
I_diffs = imgs - np.repeat(I_shadow[np.newaxis, ...], num_imgs, axis=0)

# In[3]:
# Define the horizontal and vertical unobstructed regions

v_xstart, v_ystart = 200, 700
v_xend, v_yend = 500, 1100
v_h = v_xend - v_xstart
v_w = v_yend - v_ystart

h_xstart, h_ystart = 900, 600
h_xend, h_yend = 1000, 1100
h_h = h_xend - h_xstart
h_w = h_yend - h_ystart

# Visualization
v_region_vis = patches.Rectangle(
    (v_ystart, v_xstart), v_w, v_h, linewidth=2, edgecolor='r', facecolor='none')
ax = plt.gca()
ax.add_patch(v_region_vis)

h_region_vis = patches.Rectangle(
    (h_ystart, h_xstart), h_w, h_h, linewidth=2, edgecolor='r', facecolor='none')
ax = plt.gca()
ax.add_patch(h_region_vis)
plt.imshow(imgs[0], cmap='gray')

# In[4]:
# Find the zero-crossing locations
v_locations = []
h_locations = []

for frame in I_diffs:
    v_locs = find_zero_crossings(frame, v_xstart, v_xend, v_ystart, v_yend)
    h_locs = find_zero_crossings(frame, h_xstart, h_xend, h_ystart, h_yend)
    v_locations.append(v_locs)
    h_locations.append(h_locs)

# In[5]:
# Solve for the shadow edges

h_shadow_edges = {}
v_shadow_edges = {}
vis = True
vis_freq = 20

hv_height = 800  # slipper
# hv_height = 450  # frog

for i in range(num_imgs):
    v_locs = v_locations[i].astype(np.float64)
    h_locs = h_locations[i]
    v_zeros = np.zeros(v_locs.shape[0])
    h_zeros = np.zeros(h_locs.shape[0])

    _, _, V = np.linalg.svd(v_locs, full_matrices=True)
    a_v, b_v, c_v = V[-1]

    _, _, V = np.linalg.svd(h_locs, full_matrices=True)
    a_h, b_h, c_h = V[-1]

    if a_v != 0:
        # Compute the two ends of the vertical shadow edge
        vx, vy = line_function_to_end_points(a_v, b_v, c_v, 0, hv_height)

    if a_h != 0:
        # Compute the two ends of the horizontal shadow edge
        hx, hy = line_function_to_end_points(a_h, b_h, c_h, hv_height, h - 1)

    # Plot the shadow edges
    if i % vis_freq == 0 and vis:
        print("i = ", i)
        if a_v != 0:
            plt.plot(vx, vy, color="red", linewidth=2)
        if a_h != 0:
            plt.plot(hx, hy, color="blue", linewidth=2)
        plt.imshow(imgs[i], cmap='gray')
        plt.show()

    # Store the shadow edges
    if a_v != 0 and a_h != 0:
        h_shadow_edges[i] = (a_h, b_h, c_h)
        v_shadow_edges[i] = (a_v, b_v, c_v)

# In[4]:
# Per-pixel shadow time estimation

diffs = np.diff(I_diffs, axis=0)
shadow_time = np.argmax(diffs, axis=0)
shadow_time[I_max - I_min < (35 / 255)] = 0.0
shadow_time_quantized = shadow_time // (num_imgs / 32)

plt.imshow(shadow_time, cmap='jet')
plt.show()
plt.imshow(shadow_time_quantized, cmap='jet')
plt.show()

# In[5]:
# Calibration of shadow lines
eps = 1e-10

# Load the extrinsics
ext_calib = np.load('./extrinsic_calib.npz')
R_h, t_h = ext_calib['rmat_h'], ext_calib['tvec_h']
R_v, t_v = ext_calib['rmat_v'], ext_calib['tvec_v']

int_calib = np.load('./intrinsic_calib.npz')
dist = int_calib['dist'].astype(np.float32)
mtx = int_calib['mtx'].astype(np.float32)

cam_c = np.zeros((3, 1), dtype=np.float32)
hcam_w = cam2world(cam_c, R_h, t_h)
vcam_w = cam2world(cam_c, R_v, t_v)

p0 = np.array([1.0, 1.0, 0], dtype=np.float32)
n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

h_inters = {}
for frame_idx, h_edge in h_shadow_edges.items():
    y1, y2 = 450, 700

    # Compute the intersections in the horizontal plane coordinate
    h_inter1, h_inter2 = camera2plane(
        frame_idx, h_edge, y1, y2, mtx, dist, R_h, t_h, hcam_w, p0, n)

    # Convert the intersections back to camera coordinate
    h_inter1 = world2cam(h_inter1, R_h, t_h)
    h_inter2 = world2cam(h_inter2, R_h, t_h)
    h_inters[frame_idx] = np.array([h_inter1, h_inter2])

v_inters = {}
for frame_idx, v_edge in v_shadow_edges.items():
    y1, y2 = 100, 450

    # Compute the intersections in the vertical plane coordinate
    v_inter1, v_inter2 = camera2plane(
        frame_idx, v_edge, y1, y2, mtx, dist, R_v, t_v, vcam_w, p0, n)

    # Convert the intersections back to camera coordinate
    v_inter1 = world2cam(v_inter1, R_v, t_v)
    v_inter2 = world2cam(v_inter2, R_v, t_v)

    v_inters[frame_idx] = np.array([v_inter1, v_inter2])

# In[5]
# Save an npz file
reconstructed_points = {}
for frame_idx in v_shadow_edges.keys():
    P1, P2 = h_inters[frame_idx]
    P3, P4 = v_inters[frame_idx]
    reconstructed_points[frame_idx] = np.array([P1, P2, P3, P4])

reconstructed_points = np.array(reconstructed_points, dtype=object)
np.savez('./recon.npz', points=reconstructed_points)

# In[5]:
# read the npz file
recon = np.load('./recon.npz', allow_pickle=True)
pts = recon['points'].item()
print(pts[55])

# In[6]:
# Test: plot the horizontal and vertical shadow lines in 3D
cnt = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for frame_idx in h_shadow_edges.keys():
    h1, h2 = h_inters[frame_idx]
    v1, v2 = v_inters[frame_idx]

    # Plot the horizontal line
    ax.plot([h1[0], h2[0]], [h1[1], h2[1]], [h1[2], h2[2]])
    # Plot the vertical line
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]])

    cnt += 1
    if cnt > 10:
        break

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
set_axes_equal(ax)

plt.show()


# In[6]:
# Calibration of shadow planes
shadow_planes = {}
for frame_idx in h_shadow_edges.keys():
    p1, p2 = h_inters[frame_idx][0], h_inters[frame_idx][1]
    p3, p4 = v_inters[frame_idx][0], v_inters[frame_idx][1]

    n = np.cross((p2 - p1), (p4 - p3))
    n /= np.linalg.norm(n)

    shadow_planes[frame_idx] = np.array([p1, n])

# In[6]
# Save an npz file
np.savez('./shadow_plane.npz',
         shadow_planes=np.array(shadow_planes, dtype=object))

# In[7]:
# Visualize shadow planes
cnt = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for frame_idx in h_shadow_edges.keys():

    # Plot the shadow edges
    h1, h2 = h_inters[frame_idx]
    v1, v2 = v_inters[frame_idx]

    # Plot the horizontal line
    ax.plot([h1[0], h2[0]], [h1[1], h2[1]], [h1[2], h2[2]])
    # Plot the vertical line
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]])

    # a plane is ax + by + cz + d = 0
    # [a, b, c] is the normal, we need to calculate d
    p, n = shadow_planes[frame_idx]
    d = -p.dot(n)

    xx, yy = np.meshgrid(range(-500, 0), range(-300, 200))
    z = (-n[0] * xx - n[1] * yy - d) * 1.0 / n[2]

    ax.plot_surface(xx, yy, z)

    break

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
set_axes_equal(ax)

plt.show()

# In[7]:
# Crop the part of the image you want to reconstruct
obj_xstart, obj_ystart = 300, 300
obj_xend, obj_yend = 650, 820
obj_h = obj_xend - obj_xstart
obj_w = obj_yend - obj_ystart

obj_region_vis = patches.Rectangle(
    (obj_ystart, obj_xstart), obj_w, obj_h, linewidth=2, edgecolor='r', facecolor='none')
ax = plt.gca()
ax.add_patch(obj_region_vis)
plt.imshow(imgs[0], cmap='gray')

# In[8]:
# Recover the depth of each pixel in the cropped image
cam_c = np.zeros(3)
pts = []
colors = []
rgb_colors = []
for x in range(obj_xstart, obj_xend):
    for y in range(obj_ystart, obj_yend):
        # Eliminate the pixels that are never shadowed by the stick
        if (I_max[x, y] - I_min[x, y]) < 70.0 / 255.0:
            continue
        point = np.array([y, x]).astype(np.float32)

        # Backproject the pixel p into a 3D ray r,
        # note that the ray is under the camera coordinate
        ray = pixel2ray(point, mtx, dist).reshape(-1)

        # Get the shadow time of the pixel
        frame_idx = shadow_time[x, y]

        # Get the shadow plane of the pixel
        if frame_idx in shadow_planes.keys():
            p1, n = shadow_planes[frame_idx]

            # Intersect the ray with the shadow plane
            inter = compute_intersection(cam_c, ray, p1, n)

            pts.append(inter)
            colors.append(imgs[frame_idx, x, y])
            rgb_colors.append(color_imgs[frame_idx, x, y, :])

pts = np.array(pts)
colors = np.array(colors)
rgb_colors = np.array(rgb_colors)
print("pts shape: ", pts.shape)
print("colors shape: ", colors.shape)
print("rgb_colors shape: ", rgb_colors.shape)

# Save npz file
np.savez('./reconstructed_points.npz', points=pts, colors=rgb_colors)

# In[9]:
# Visualize the point cloud
print(pts[:, 2].max())
print(pts[:, 2].min())

mask = (np.linalg.norm(pts, axis=1) < 2000).astype(bool)
mask = mask & ((np.linalg.norm(pts, axis=1) > 1750).astype(bool))

pts_filtered = pts[mask]
colors_filtered = colors[mask]
rgb_colors_filtered = rgb_colors[mask]

pts_small = pts[::5, :]
colors_small = colors[::5]
rgb_colors_small = rgb_colors[::5, :]

# In[9]:
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

s = np.ones_like(colors) * 0.01
ax.scatter(pts_small[:, 0], pts_small[:, 1], pts_small[:, 2],
           s=0.05, c=colors_small, cmap='gray')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
set_axes_equal(ax)

plt.show()

# In[10]:
# Visualize using open3d
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(pts_filtered)
pc.colors = o3d.utility.Vector3dVector(rgb_colors_filtered)
o3d.visualization.draw_geometries([pc])

# %%
