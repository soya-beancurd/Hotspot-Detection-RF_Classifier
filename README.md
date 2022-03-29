# Hotspot Detection with Random Forest Classifier and Google Earth Engine
 Source Code and Documentation

This repository primarily contains all source codes and corresponding documentation for the study in which this GitHub repository link was provided in its Appendix.

A total of 3 main files will be maintained in this repository:
1) **Data.zip** - A zipped folder of the datasets used for this study. Each 'image' will have a corresponding raster file (*.tif*) and set of vector shape files (*.shp*). An additional excel file (*.xlsx*) detailing the list of datasets (name, coordinates, dates, etc) used for this study is also included in this zipped folder. It is this excel file that the following codes (*Model.ipynb*) will read to extract the corresponding dataset when attempting to run the algorithm.

2) **GEE.ipynb** - An interactive python 3 file (created on Jupyter Notebook) that contains code for GEE image extraction, labelling and subsequent exportation. 
- The python 3 libraries employed in this code is:
  - ee
  - geemap
  - pandas
  - os

3) **Model.ipynb** - An interactive python 3 file (created on Jupyter Notebook) that contains code that extracts relevant data from the *Data* folder, trains the model, and plot/save the model's output. 
- The python 3 libraries employed in this code is:
  - numpy
  - sklearn

4) **utility_functions.py** - A python 3 file (created on Spyder) that contains all utility functions used in *Model.ipynb*.
- The python 3 libraries employed in this code is:
  - copy
  - geopandas
  - matplotlib
  - numpy
  - os
  - pandas
  - rasterio
  - seaborn
  - sklearn
