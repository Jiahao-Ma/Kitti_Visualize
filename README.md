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

![image](Kitti_Visualize\git_image\download_link.png)

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