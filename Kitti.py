import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import Calib, Object3D, roty
from PIL import Image
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
"""  Calibration Information
    3d XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.

    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
    x_ref = Tr_velo_to_cam * x_velo
    x_rect = R0_rect * x_ref

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]
                = K * [1|t]

    image2 coord:
        ----> x-axis (u)
    |
    |
    v y-axis (v)

    Velodyne coordinate system:
    front x, left y, up z

    #     7 -------- 4
    #    /|  (z)    /|
    #   6 ---|---- 5 .
    #   | |  |     | |
    #   . 3 -|------ 0
    #   |/   .- - -|/ - - -> (y)
    #   2 --/------- 1
    #      /
    #     /
    #   (x)

    rect/ref camera coord:
    right x, down y, front z

    #     7 -------- 4
    #    /|         /|
    #   6 -------- 5 .
    #   | |        | |
    #   . 3 -------- 0
    #   |/   .- - -|/ - - -> (x)
    #   2 ---|----- 1
    #        |
    #        | (y)

    Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
"""

class Kitti_Dataset(object):
    def __init__(self, base_path, mode='training') -> None:
        super(Kitti_Dataset, self).__init__()
        self.dir_path = os.path.join(base_path, mode)
        self.calib_path = os.path.join(self.dir_path, 'calib')
        self.img_path = os.path.join(self.dir_path, 'image_2')
        self.label_path = os.path.join(self.dir_path, 'label_2')
        self.pc_path = os.path.join(self.dir_path, 'velodyne')
        self.IDList = [ p.split('.')[0] for p in sorted(os.listdir(self.img_path))]
    def __len__(self):
        return len(os.listdir(self.img_path))
    
    def get_calib(self, index):
        calib_path = os.path.join(self.calib_path, self.IDList[index]+'.txt')
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines if len(l) and l!='\n']
        dict_calib = dict()
        for line in lines:
            key, value = line.split(':')
            dict_calib[key] = np.array([float(x) for x in value.split()])
        return Calib(dict_calib)
    
    def get_rgb(self, index):
        rgb_path = os.path.join(self.img_path, self.IDList[index]+'.png')
        # return cv2.imread(rgb_path)
        return Image.open(rgb_path)
    
    def get_pcs(self, index):
        pc_path = os.path.join(self.pc_path, self.IDList[index]+'.bin')
        return np.fromfile(pc_path, dtype=np.float32, count=-1).reshape((-1, 4))

    def get_label(self, index):
        label_path = os.path.join(self.label_path, self.IDList[index]+'.txt')
        with open(label_path) as f:
            lines = f.readlines()
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))
        return [Object3D(x) for x in lines]


"""
#############################
#   Kitti Visualization     #
#   Draw 2/3D bbox in rgb   #
#############################
"""   

def draw_2DBBox_in_rgb(img, labels, show=True, linewidth=3, edgecolor=(0, 1, 0), fontcolor=(0, 1, 0.5), fontsize=12, figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for obj in labels:
        width = obj.bbox[2] - obj.bbox[0]
        height = obj.bbox[3] - obj.bbox[1]
        ax.add_patch(patches.Rectangle((obj.bbox[0], obj.bbox[1]), width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor='none'))
        ax.text(obj.bbox[0]-5, obj.bbox[1]-5, s=obj.name, c=fontcolor, fontsize=fontsize)
    ax.imshow(img)
    if show:
        plt.show()
    return ax

# Camera coordinate system
#     7 -------- 4
#    /|         /|
#   6 -------- 5 .
#   | |        | |
#   . 3 -------- 0
#   |/   .- - -|/ - - -> (x)
#   2 ---|----- 1
#        |
#        | (y)

def project_to_image(corner_3d, calib:Calib):
    """ Project corner_3d `nx4 points` in camera rect coord to image2 plane
        Args:
            corner_3d: nx4 `numpy.ndarray`
            calib: `Calib` calib.P2: camera2 projection matrix
        Returns:
            corner_2d: nx2 `numpy.ndarray` 2d points in image2
    """
    corner_2d = np.dot(calib.P2, corner_3d.T)
    corner_2d[0, :] = corner_2d[0, :] / corner_2d[2, :]
    corner_2d[1, :] = corner_2d[1, :] / corner_2d[2, :]
    corner_2d = np.array(corner_2d, dtype=np.int)
    return corner_2d[0:2, :].T

def compute_3d_bbox(obj:Object3D, calib:Calib):
    """ obj.dimension and obj.location are the width, height, length 
        and bottom center of 3D bbox in `rect camera coord`!
        Retunrs:
            corner_2d: (8, 2) array in left image coord
            corner_3d: (8, 3) array in rect camera coord
    """
    # 3d bbox dimension: height, width and length
    h, w, l = obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]
    # 3d bbox corners
    x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y = [0., 0., 0., 0., -h, -h, -h, -h]
    z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # compute rotation matrix around yaw axis
    rotMat = roty(obj.rotation_y)
    corner_3d = np.vstack([x, y, z])
    # rotate and translate the 3d bbox
    corner_3d = np.dot(rotMat, corner_3d)
    bottom_center = np.tile(obj.location, (8, 1)).T
    corner_3d = corner_3d + bottom_center
    corner_3d_homo = np.vstack([corner_3d, np.ones((1, corner_3d.shape[1]))])
    # project corner_3d in rect camera coord to image2 plane
    corner_2d = project_to_image(corner_3d_homo.T, calib)
    
    return corner_2d, corner_3d.T

def draw_3DBBox_in_rgb(img, labels, calib, show=True, linewidth=2, color=(0, 1, 0), fontcolor=(0, 1, 0.5), fontsize=12, figsize=(10,10)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for obj in labels:
        # （8, 2） 2d points in image2 coord
        corner_2d_rect, _ = compute_3d_bbox(obj, calib)
        ax.text(corner_2d_rect[-1, 0]-5, corner_2d_rect[-1, 1]-5, s=obj.name, c=fontcolor, fontsize=fontsize)
        ax = draw_3DBBox(ax, corner_2d_rect, linewidth=linewidth, edgecolor=color)
    ax.imshow(img)
    if show:
        plt.show()
    return ax

"""
######################################
#   Kitti Visualization              #
#   Draw 3D bbox in velodyne coord   #
######################################
""" 
#  Velodyne coordinate system
#     7 -------- 4
#    /|  (z)    /|
#   6 ---|---- 5 .
#   | |  |     | |
#   . 3 -|------ 0
#   |/   .- - -|/ - - -> (y)
#   2 --/------- 1
#      /
#     /
#   (x)

def draw_3DBBox_in_velo(pcs_3d_in_velo, labels, calib:Calib, linewidth=2, edgecolor=(0, 1, 0), show=True, xlim3d=None, ylim3d=None, zlim3d=None, figsize=(10,10)):
    color = ['blue', 'red', 'green', 'yellow', 'black', 'white', 'purple', 'salmon']
    ax = draw_pointclouds_in_velo(pcs_3d_in_velo, show=False, xlim3d=xlim3d, ylim3d=ylim3d, zlim3d=zlim3d, figsize=figsize)
    for obj in labels:
        _, corner_3d_rect = compute_3d_bbox(obj, calib)
        corner_3d_velo = calib.project_rect_to_velo(corner_3d_rect)
        # corner_3d_velo: (8, 3) 3d points in velodyne coord.
        # corner_3d_velo = corner_3d_velo[::-1]
        ax = draw_3DBBox(ax, corner_3d_velo, linewidth=linewidth, edgecolor=edgecolor)
        ax.scatter(corner_3d_velo[:, 0], corner_3d_velo[:, 1], corner_3d_velo[:,2], s=3, c=color)
        
    if show:
        plt.show()
    return ax

def draw_3DBBox(ax, pts, linewidth=2, edgecolor=(0, 1, 0)):
    assert ax != None, 'ax has not been created !'
    assert pts.shape[1] in [2, 3], 'The shape of input point must be nx2 or nx3 !'
    if pts.shape[1] == 2:
        # 2d image2 coord
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            ax.plot((pts[i, 0], pts[j, 0]), (pts[i, 1], pts[j, 1]), color=edgecolor, linewidth=linewidth)
            i, j = k + 4, (k + 1) % 4 + 4
            ax.plot((pts[i, 0], pts[j, 0]), (pts[i, 1], pts[j, 1]), color=edgecolor, linewidth=linewidth)
            i, j = k, k + 4
            ax.plot((pts[i, 0], pts[j, 0]), (pts[i, 1], pts[j, 1]), color=edgecolor, linewidth=linewidth)
    elif pts.shape[1] == 3:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            ax.plot((pts[i, 0], pts[j, 0]), (pts[i, 1], pts[j, 1]), (pts[i, 2], pts[j, 2]), color=edgecolor, linewidth=linewidth)
            i, j = k + 4, (k + 1) % 4 + 4
            ax.plot((pts[i, 0], pts[j, 0]), (pts[i, 1], pts[j, 1]), (pts[i, 2], pts[j, 2]), color=edgecolor, linewidth=linewidth)
            i, j = k, k + 4
            ax.plot((pts[i, 0], pts[j, 0]), (pts[i, 1], pts[j, 1]), (pts[i, 2], pts[j, 2]), color=edgecolor, linewidth=linewidth)

    return ax

"""
###########################################
#   Kitti Visualization                   #
#   Draw point clouds in velodyne coord   #
###########################################
""" 
def draw_pointclouds_in_velo(point_clouds, points=0.2, axes=[0,1,2], xlim3d=None, ylim3d=None, zlim3d=None, show=True, figsize=(10,10)) -> Axes3D:
    # visualization limits
    axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
    ]
    axes_str = ['X', 'Y', 'Z']
    point_size = 0.01 * (1. / points)
    point_step = int(1. / points)
    velo_range = range(0, len(point_clouds), point_step)
    point_clouds = point_clouds[velo_range]
    print('point_size: {:.2f}  point_step: {}  total_point_num: {}'.format(point_size, point_step, len(point_clouds)))
    # color 
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3]
    depth = point_clouds[:, 0]
    color = np.clip(np.array(640 / depth, np.int), a_min=0, a_max=255)
    xs = point_clouds[:, 0]
    ys = point_clouds[:, 1]
    zs = point_clouds[:, 2]

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    # ax = Axes3D(fig)
    ax.scatter(xs, ys, zs, s = point_size, c = cmap[color, :])

    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)
    ax.view_init(elev=40, azim=180)
    
    if show:
        plt.show()
    return ax

"""
########################################
#   Kitti Visualization                #
#   Draw point clouds in image2 plane  #
########################################
""" 
def draw_pointclouds_in_rgb(pcs_3D_velo, img, calib:Calib, clip_distance=5, radius=1, thickness=1, show=True, figsize=(10,10)):
    """
        Args:
            pcs_3D_velo: nx3 points in velodyne coord.
            img: RGB image
            calib: `Calib` contains the calibration information 
                 and the projection method ( 2D <-> 3D )
            clip_distance: only show the point clouds'x corrdinate value larger than clip_distance

            #  Velodyne coordinate system
            #     7 -------- 4
            #    /|  (z)    /|
            #   6 ---|---- 5 .
            #   | |  |     | |
            #   . 3 -|------ 0
            #   |/   .- - -|/ - - -> (y)
            #   2 --/------- 1
            #      /
            #     /
            # ----------  clip_distance
            #    /
            #  (x)

    """
    if pcs_3D_velo.shape[1] == 4:
        # x, y, z, reflection
        pcs_3D_velo = pcs_3D_velo[:, :3]
    pts_2D = calib.project_velo_to_image(pcs_3D_velo)
    # clamp 
    height, width, c = np.array(img).shape
    xmin, ymin, xmax, ymax = 0, 0, width, height
    keep = (
        (pts_2D[:, 0] < xmax) &
        (pts_2D[:, 0] >= xmin) &
        (pts_2D[:, 1] < ymax) &
        (pts_2D[:, 1] >= ymin) 
    ) & (pcs_3D_velo[:, 0] > clip_distance)
    imgfov_pts_2D = pts_2D[keep]
    pcs_3D_velo = pcs_3D_velo[keep]
    imgfov_pc_rect = calib.project_velo_to_rect(pcs_3D_velo)
    

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3]
    depth = imgfov_pc_rect[:, 2]
    color = cmap[np.array(640 / depth, np.int), :]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.scatter(
        np.array(np.round(imgfov_pts_2D[:, 0]), np.int), 
        np.array(np.round(imgfov_pts_2D[:, 1]), np.int),
        radius,
        c=color,
    )
    ax.imshow(img)
    if show:
        plt.show()
    return ax


if __name__ == "__main__":
    import copy
    base_path = r'F:\ObjectDetection\dataset\KITTI\Kitti'
    kitti = Kitti_Dataset(base_path)
    index = 2
    img = kitti.get_rgb(index)
    img_3d = copy.deepcopy(img)
    labels = kitti.get_label(index)
    calib = kitti.get_calib(index)
    point_clouds = kitti.get_pcs(index)
    draw_2DBBox_in_rgb(img, labels)
    draw_3DBBox_in_rgb(img_3d, labels, calib)
    draw_pointclouds_in_velo(point_clouds, xlim3d=[-40, 40])
    draw_pointclouds_in_rgb(point_clouds, img, calib)
    draw_3DBBox_in_velo(point_clouds, labels, calib, xlim3d=[-40, 40])

