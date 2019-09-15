# About
This project experiments with the 11k Hands dataset by Mahmoud Afifi. We use
Color Moments and SIFT as features and implement distance functions to compare
two images.

# Requirements:
1. Python 3.7.4
2. MongoDB
3. lib32-glib2 to run SIFT executable from http://www.cs.ubc.ca/~lowe/keypoints/

The SIFT executable from David Lowe's page takes about 3 seconds to calculate
keypoints for one image. If you would like to use OpenCV's implementation, you
need to recompile OpenCV from source using the option `OPENCV_ENABLE_NONFREE`.

To do that, clone the opencv and opencv contrib repositories. Then run
```
$ cmake <path to opencv clone> \
-D CMAKE_BUILD_TYPE=RELEASE \
-D BUILD_PYTHON_SUPPORT=ON \
-D WITH_XINE=ON \
-D WITH_OPENGL=ON \
-D WITH_TBB=ON \
-D BUILD_EXAMPLES=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D WITH_V4L=ON \
-D CMAKE_INSTALL_PREFIX=<install path> \
-D OPENCV_EXTRA_MODULES_PATH=<oprncv_contrib clone path> \
-D OPENCV_ENABLE_NONFREE=ON
```

If you specify `CMAKE_INSTALL_PREFIX`, you need to export this install path to
`PYTHONPATH` using `export PYTHONPATH=~/.opencv/lib/python3.7/site-packages/`.

# Python libraries used
1. yapf
2. tqdm
3. numpy
4. opencv-python
5. Jinja2
6. dynaconf
7. pymongo
8. pickle

Just run pip install -r requirements.txt

# Configuration
The configuration can be edited in `settings.toml`. There are comments in the
file describing each of the configurations. To get this program running, it is
enough to provide `username`, `password`, `host`, `port` and `data_path`.

# Execution
To build the feature database, run:
```
$ python db_make.py -m <model>
```

To visualize or check the raw features of an image, run:
```
$ python feature.py -m <model> -i <image path> -v
```

To find top _k_ similar images, run:
```
$ python similarity.py -m <model> -i <image path> -k <number of matches required>
```
