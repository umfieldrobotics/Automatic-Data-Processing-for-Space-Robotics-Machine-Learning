# Automatic-Data-Processing-for-Space-Robotics-Machine-Learning

Welcome to the repo for the paper titled "Automatic Data Processing for Space Robotics Machine Learning"! The paper can be found here: https://arxiv.org/abs/2310.01932

This repo contains PyQGIS code for generating viewsheds of the Curiosity or Perseverance rovers' points of view from a given NASA PDS image and label.

## conda setup
```
$ conda create --name automatic-data-processing python=3.10.12
$ conda activate automatic-data-processing
$ conda install qgis --channel conda-forge
$ conda install -c conda-forge ocl-icd-system
$ pip install pvl pdr
$ pip install opencv-python-headless
```

## File setup
First, clone the repo.

Please download the MSL and Mars2020 DEM files from this folder and place them into the `qgis` directory in your cloned repo: https://drive.google.com/drive/folders/1GmYnekSMn2mPa3q1FcFp7Cfd1eF9q1bA?usp=sharing

Place the Mastcam images that you'd like to have processed into `qgis/msl_images` and labels into `qgis/msl_labels` for Curiosity rover images, and `qgis/mars2020_images` and `qgis/mars2020_labels` for the Perseverance rover images. Make sure that the images you pull from NASA are MastCam or MastCamZ, and not another camera!

## Running the code
In the `qgis_pipeline.py` script, first check that the `QGIS_PYTHON_INSTALL` path and the `WORKSPACE_PATH` are correct. Then, in the main function, make sure that `MISSION` is representative of your desired mission: either `MSL()` or `MARS2020()`.

Then, run the code by executing the following in your conda environment: `python3 qgis_pipeline.py`.

To see the generated viewsheds, open up QGIS and open the `base_msl.qgz` or `base_mars2020.qgz` (depending on whether you are working with Curiosity or Perseverance images.
