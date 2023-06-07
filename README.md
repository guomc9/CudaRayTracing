# CudaRayTracing
A simple ray-tracing program implemented with CUDA, which supports rendering images and videos.

<img src="models/Scene-03/render/scene-01.gif" style="zoom:70%;" />



<img src="models\CornellBox\scene-msaa.png" style="zoom:70%;" />

## Getting Started

These instructions will provide you with a copy of the project running on your local machine for development and rendering. See the deployment section for notes on how to deploy the project in a live environment.

### Prerequisites

Before getting started, make sure you have the following requirements:

- OpenCV(4.x or above). For Windows machines, you should build OpenCV with MSVC in DEBUG mode to get `opencv_world*d.lib` and `opencv_videoio*d.lib` with the option `BUILD_opencv_world` on and off. For Unix-like environments, you can build OpenCV using the regular steps.
-  CUDA

### Installation

Follow these steps to build the project locally:

1. Clone the repository: 

   ```shell
   git clone https://github.com/guomingce/CudaRayTracing.git
   ```

2.  Build the program:

   ```shell
   mkdir build
   cd build
   cmake build ..
   ```

### Configure Rendering Task

You can configure your rendering task with the `./config.json` file, in which you can set arguments for ray-tracing rendering.

You can see examples below. 

For `task-1`, it is a image rendering task, you can set your `.obj` file path  and `.mtl` directory, as for the loader implemented with `tiny_obj_loader`, then `eye_pos`, `fovY` , `width`, `height` denote the camera position, vertical field of view, image width and height, and ray-tracing parameters including BVH nodes' triangles threshold, possibility Russian Roulette, sample per pixel,  and number of light samples. 

For `task-2`, it is a video rendering task, in addition to the relevant parameters before, you are required to provide the rotation axis, angles(degree), speed(degree per frame), frames per second.

 ```json
 {
     "tasks": [
         {
             "task_id": 1,
             "task_type": "I",
             "OBJ_paths": [
                 {
                     "OBJ_path": "../../models/Scene-01/scene.obj",
                     "MTL_dir": "../../models/Scene-01/"
                 }
             ],
             "eye_pos": {
                 "x": 0.0, 
                 "y": 60.0, 
                 "z":-330.0
             },
             "fovY": 45.0,
             "width": 1024,
             "height": 512,
             "bvh_thresh_n": 8,
             "P_RR": 0.99,
             "spp": 64,
             "light_sample_n": 32,
             "image_save_path": "../../models/Scene-01/scene.png"
         },
         {
             "task_id": 2, 
             "task_type": "V",
             "OBJ_paths": [
                 {
                     "OBJ_path": "../../models/Scene-03/scene.obj",
                     "MTL_dir": "../../models/Scene-03/"
                 }
             ],
             "eye_pos": {
                 "x": 0.0, 
                 "y": 60.0, 
                 "z":-330.0
             },
             "fovY": 45.0,
             "width": 1024,
             "height": 512,
             "bvh_thresh_n": 8,
             "P_RR": 0.99,
             "spp": 64,
             "light_sample_n": 32,
             "rot_axis": {
                 "x": 0.0, 
                 "y": 1.0, 
                 "z": 0.0
             },
             "rot_angle": 30,
             "rot_speed": 1,
             "fps":10,
             "video_save_path": "../../models/Scene-03/render/scene.avi"
         }
     ]
 }
 ```

Besides, you can also make a images dataset for NeRF training with this program. As the `task-3` below, after set the rotation parameters, transformation save path  and image save directory, you can run the program to get your data.

```json
{
    "tasks": [
        {
            "task_id": 3, 
            "task_type": "RD",
            "OBJ_paths": [
                {
                    "OBJ_path": "../../models/Scene-04/scene.obj",
                    "MTL_dir": "../../models/Scene-04/"
                }
            ],
            "eye_pos": {
                "x": 0.0, 
                "y": 60.0, 
                "z": -380.0
            },
            "fovY": 45.0,
            "width": 1024,
            "height": 512,
            "bvh_thresh_n": 4,
            "P_RR": 0.9,
            "spp": 32,
            "light_sample_n": 32,
            "rot_axis": {
                "x": 0.0, 
                "y": 1.0, 
                "z": 0.0
            },
            "rot_angle": 3,
            "rot_speed": 1,
            "transform_save_path": "../../models/Scene-04/data/transform.json",
            "image_save_dir": "../../models/Scene-04/data/"
        }
    ]
}
```

Please notice that the executable file is in the directory ` ./build/` for Unix-like environments, so you are supposed to fix the save paths or directories.