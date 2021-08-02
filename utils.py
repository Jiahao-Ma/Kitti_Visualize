import numpy as np

class Object3D(object):
    def __init__(self, content) -> None:
        super(Object3D, self).__init__()
        lines = content.split()
        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(lines[3])
        self.bbox = np.array([float(lines[4]), float(lines[5]), float(lines[6]), float(lines[7])])
        self.dimensions = np.array([float(lines[8]), float(lines[9]), float(lines[10])])
        self.location = np.array([float(lines[11]), float(lines[12]), float(lines[13])])
        self.rotation_y = float(lines[14])


class Calib(object):
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
    def __init__(self, dict_calib) -> None:
        super(Calib, self).__init__()
        self.P0 = dict_calib['P0'].reshape((3, 4))
        self.P1 = dict_calib['P1'].reshape((3, 4))
        self.P2 = dict_calib['P2'].reshape((3, 4))
        self.P3 = dict_calib['P3'].reshape((3, 4))
        self.R0_rect = dict_calib['R0_rect'].reshape((3,3))
        self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape((3, 4))
        self.Tr_cam_to_velo = inverse_transformation_matrix(self.Tr_velo_to_cam)
        self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape((3, 4))

    def __str__(self):
        np.set_printoptions(precision=3, suppress=True)
        calib_info = """ P0: \n{}\nP1: \n{}\nP2: \n{}\nP3: \n{}\nTr_velo_to_cam: \n{}\nTr_cam_to_velo: \n{}\nTr_imu_to_velo: \n{}"""\
                     .format(self.P0.round(2), 
                            self.P1.round(2), 
                            self.P2.round(2), 
                            self.P3.round(2),
                            self.Tr_velo_to_cam.round(2),
                            self.Tr_cam_to_velo.round(2),
                            self.Tr_imu_to_velo.round(2))
        return calib_info
        
    # ===========================
    # ------- 3d to 2d ----------
    # ---- velodyne to image2 ---
    # ===========================
    def cart2hom(self, pcs_3d):
        """ Input: nx3 points in Cartesian in Velodyne coordinate system
            Output: nx4 points in Homogeneous by pending 1
        """
        n = pcs_3d.shape[0]
        pcs_3d_hom = np.hstack([pcs_3d, np.ones((n, 1))])
        return pcs_3d_hom
    
    def project_velo_to_ref(self, pcs_3d_velo):
        """ Input and output are nx3 points"""
        pcs_3d_velo = self.cart2hom(pcs_3d_velo) # nx4 points
        # 3x4 @ 4xn = 3xn -> nx3
        return np.dot(self.Tr_velo_to_cam, pcs_3d_velo.T).T

    def project_ref_to_rect(self, pcs_3d_ref):
        """ Input and output are nx3 points"""
        # 3x3 @ 3xn = 3xn -> nx3
        return np.dot(self.R0_rect, pcs_3d_ref.T).T
    
    
    def project_velo_to_rect(self, pcs_3d_velo):
        """ Input and output are nx3 points"""
        pcs_3d_ref = self.project_velo_to_ref(pcs_3d_velo)
        return self.project_ref_to_rect(pcs_3d_ref)

    def project_rect_to_image(self, pcs_3d_rect):
        """ Input are nx3 points in rect camera coord.
            Output are nx2 points in image2 coord.
        """
        # nx3 -> nx4
        pcs_3d_rect = self.cart2hom(pcs_3d_rect)
        # 3x4 @ 4xn = 3xn
        pts_2d = np.dot(self.P2, pcs_3d_rect.T)
        pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
        pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
        # 3xn -> nx3
        pts_2d = pts_2d.T
        return pts_2d[:, :2]

    def project_velo_to_image(self, pcs_3d_velo):
        """ Input are nx3 points in velodyne coord.
            Output are nx2 points in image2 coord.
        """
        pcs_3d_rect = self.project_velo_to_rect(pcs_3d_velo)
        return self.project_rect_to_image(pcs_3d_rect)


    # ===========================
    # ------- 3d to 3d ----------
    # ---- image2 to velodyne ---
    # ===========================
    def project_rect_to_velo(self, pcs_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pcs_3d_ref = self.project_rect_to_ref(pcs_3d_rect)
        return self.project_ref_to_velo(pcs_3d_ref)
        

    def project_rect_to_ref(self, pcs_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in ref camera coord.
        """
        # 3x3 @ 3xn = 3xn -> nx3
        return np.dot(np.linalg.inv(self.R0_rect), pcs_3d_rect.T).T
    
    def project_ref_to_velo(self, pcs_3d_ref):
        """ Input: nx3 points in ref camera coord.
            Output: nx3 points in velodyne coord.
        """
        # nx3 -> nx4
        pcs_3d_ref = self.cart2hom(pcs_3d_ref)
        # nx4 @ 4x3 = nx3
        return np.dot(pcs_3d_ref, self.Tr_cam_to_velo.T)



def rotx(t):
    """ Rotation about x-axis """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(t):
    """ Rotation about y-axis """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(t):
    """ Rotation about z-axis """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def inverse_transformation_matrix(Tr):
    """ Inverse a rigid body transformation matrix (3x4)
    """
    inv_Tr = np.zeros_like(Tr)
    inv_Tr[:, :3] = Tr[:, :3].T
    inv_Tr[:, 3] = np.dot(-Tr[:, :3].T, Tr[:, 3])
    return inv_Tr # 3x4


if __name__ == '__main__':
    from Kitti import *
    import copy
    base_path = r'F:\ObjectDetection\dataset\KITTI\Kitti'
    kitti = Kitti_Dataset(base_path)
    index = 0
    img = kitti.get_rgb(index)
    img_3d = copy.deepcopy(img)
    labels = kitti.get_label(index)
    calib = kitti.get_calib(index)
    point_clouds = kitti.get_pcs(index)
    for obj in labels:
        pts_3d_rect = obj.location
        print('Target bottom center in rect camera coord: \n', pts_3d_rect)
        pts_3d_ref = calib.project_rect_to_ref(pts_3d_rect)
        print('\nProject 3d point from rect camera coord to ref camera coord:\n', pts_3d_ref)
        pts_3d_velo = calib.project_ref_to_velo(pts_3d_ref[None, :])
        print('\nProject 3d point from ref camera coord to velodyne coord:\n', pts_3d_velo)
        inverse_3d_ref = calib.project_velo_to_ref(pts_3d_velo)
        print('\nProject 3d point from velodyne coord to ref camera coord to:\n', inverse_3d_ref)
