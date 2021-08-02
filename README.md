# Kitti_Visualize
 
## Dependencies

Common dependencies like `numpy`, `matplotlib`, `PIL`

## Project structure

| File                   | Description                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `kitti_demo_notebook.ipynb`  | Jupyter Notebook with dataset visualisation routines and output.                                 |
| `Kitti.py`  | Methods for parsing labels, calibration information, image and lidar point clouds.  |
| `utils.py`         | Method for projecting 3D to 2D and its invers transformation.                                                                     |

## Dataset

Download the data (calib, image\_2, label\_2, velodyne) from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 

![image](https://github.com/Robert-Mar/Kitti_Visualize/blob/main/git_image/download_link.png)

And place it in your data folder at `Kitti`. The folder structure is as following:
```
kitti
    training
        calib
            000000.txt
        image_2
            000000.png
        label_2
            000000.txt
        velodyne
            000000.bin
        pred
            000000.txt
```

## NOTICE:
### 3d XYZ in \<label\>.txt are in rect camera coord.
### 2d box xy are in image2 coord
### Points in \<lidar\>.bin are in Velodyne coord.
***
# Coordinate Transformation:
    y_image2 = P^2_rect * x_rect
    y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
    x_ref = Tr_velo_to_cam * x_velo
    x_rect = R0_rect * x_ref

    P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                0,      0,      1,      0]
                = K * [1|t]
***
# Coordinate Visualization:
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

***