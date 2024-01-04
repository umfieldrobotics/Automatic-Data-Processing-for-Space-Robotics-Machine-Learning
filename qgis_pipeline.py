import sys
import os

QGIS_PYTHON_INSTALL = '/usr/share/qgis/python/'
WORKSPACE_PATH = os.getenv("HOME") + '/Automatic-Data-Processing-for-Space-Robotics-Machine-Learning/'

import csv
import cv2
import math
import numpy as np

from dataclasses import dataclass

import pvl
import pdr

from osgeo import gdal

sys.path.append(QGIS_PYTHON_INSTALL)                  # PyQGIS path

from qgis.core import *
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
from PyQt5.QtGui import QColor

sys.path.append(QGIS_PYTHON_INSTALL + 'plugins/')     # PyQGIS processing plugin path

import processing # 'processing' and 'ViewshedAnalysis' must be in the directory in the sys.append above (QGIS_PYTHON_INSTALL + /plugins)

from processing.core.Processing import Processing
from ViewshedAnalysis.visibility_provider import VisibilityProvider # must install Viewshed Analysis tool through QGIS plugins interface


def image_to_global(raster, x, y):
  geo_t = raster.GetGeoTransform()
  return [gdal.ApplyGeoTransform(geo_t, int(y[i]), int(x[i])) for i in range(len(x))]

# x, y in global coord
def get_elevation_at_point(layer, x, y):
  provider = layer.dataProvider()
  extent = layer.extent()

  xmin, ymin, xmax, ymax = extent.toRectF().getCoords()

  cols = layer.width()
  rows = layer.height()

  pixel_width = layer.rasterUnitsPerPixelX()
  pixel_height = layer.rasterUnitsPerPixelY()

  block = provider.block(1, extent, cols, rows)

  row = int((x - xmin) / pixel_width)
  col = int((ymax - y) / pixel_height)

  return block.value(row, col)

@dataclass
class MSL:
  IMGS_PATH = 'qgis/msl_images/'
  LBLS_PATH = 'qgis/msl_labels/'
  PROJ_PATH = 'base_msl.qgz' # base QGIS project with georeferenced aerial maps
  ROVER_PATH = 'localized_interp.csv'
  DTM_LAYER = 'msl_orbital_dem'

@dataclass
class MARS2020:
  IMGS_PATH = 'qgis/mars2020_images/'
  LBLS_PATH = 'qgis/mars2020_labels/'
  PROJ_PATH = 'base_mars2020.qgz' # base QGIS project with georeferenced aerial maps
  ROVER_PATH = 'best_interp.csv'
  DTM_LAYER = 'm20_orbital_dem'


class PDS3:
  def __init__(self, img_filepath, lbl_filepath):
    self.img_filepath = img_filepath
    self.lbl_filepath = lbl_filepath
    self.img_id = self.img_filepath.split('/')[-1][:-4] # get file name from path without .jpg
    self.img = cv2.cvtColor(cv2.imread(self.img_filepath), cv2.COLOR_BGR2RGB) # convert from BGR'
    self.lbl = pvl.load(self.lbl_filepath)

    ############################
    ### Read data from label ###
    ############################

    self.data = {}
    self.data['fixed_instrument_azimuth'] = self.lbl['DERIVED_IMAGE_PARMS']['FIXED_INSTRUMENT_AZIMUTH']
    self.data['fixed_instrument_elevation'] = self.lbl['DERIVED_IMAGE_PARMS']['FIXED_INSTRUMENT_ELEVATION']
    self.data['origin_rotation_quaternion'] = self.lbl['RSM_COORDINATE_SYSTEM_PARMS']['ORIGIN_ROTATION_QUATERNION']

    self.data['site'] = self.lbl['ROVER_MOTION_COUNTER'][0]
    self.data['drive'] = self.lbl['ROVER_MOTION_COUNTER'][1]

    self.data['c'] = np.array(self.lbl['GEOMETRIC_CAMERA_MODEL_PARMS']['MODEL_COMPONENT_1'])
    self.data['a'] = np.array(self.lbl['GEOMETRIC_CAMERA_MODEL_PARMS']['MODEL_COMPONENT_2'])
    self.data['h'] = np.array(self.lbl['GEOMETRIC_CAMERA_MODEL_PARMS']['MODEL_COMPONENT_3'])
    self.data['v'] = np.array(self.lbl['GEOMETRIC_CAMERA_MODEL_PARMS']['MODEL_COMPONENT_4'])

    self.data['horizontal_fov'] = np.array(self.lbl['INSTRUMENT_STATE_PARMS']['HORIZONTAL_FOV'])
    self.data['vertical_fov'] = np.array(self.lbl['INSTRUMENT_STATE_PARMS']['VERTICAL_FOV'])


class PDS4:
  def __init__(self, img_filepath, lbl_filepath):
    self.img_filepath = img_filepath
    self.lbl_filepath = lbl_filepath
    self.img_id = self.img_filepath.split('/')[-1][:-4] # get file name from path without .jpg
    self.img = cv2.cvtColor(cv2.imread(self.img_filepath), cv2.COLOR_BGR2RGB) # convert from BGR
    self.lbl = pdr.read(self.lbl_filepath) # returns lxml ElementTree

    self.data = {}
    # reference for what different items in the XML PDS4 label file are: https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_mission/document_camera/Mars2020_Camera_SIS_Labels_sort_vicar.html 
    self.data['fixed_instrument_azimuth'] = float(self.lbl['label'][1][7][1][1][16].find('geom:instrument_azimuth').text) # deg, Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Derived_Geometry[*]/instrument_azimuth
    self.data['fixed_instrument_elevation'] = float(self.lbl['label'][1][7][1][1][16].find('geom:instrument_elevation').text) # deg, Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Derived_Geometry[*]/instrument_elevation
    # self.data['origin_rotation_quaternion'] =

    self.data['site'] = int(self.lbl['label'][1][7][1][1][17][1].find('geom:index_value_number').text) # SITE, Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Motion_Counter/Motion_Counter_Index[*]/index_value_number
    self.data['drive'] = int(self.lbl['label'][1][7][1][1][17][2].find('geom:index_value_number').text) # DRIVE, Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Motion_Counter/Motion_Counter_Index[*]/index_value_number

    self.data['c'] = np.array([self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Center')[0].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Center')[1].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Center')[2].text]) # m, Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Camera_Model_Parameters/CAHVOR_Model/Vector_Center
    self.data['a'] = np.array([self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Axis')[0].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Axis')[1].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Axis')[2].text]) # Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Camera_Model_Parameters/CAHVOR_Model/Vector_Axis
    self.data['h'] = np.array([self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Horizontal')[0].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Horizontal')[1].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Horizontal')[2].text]) # Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Camera_Model_Parameters/CAHVOR_Model/Vector_Horizontal
    self.data['v'] = np.array([self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Vertical')[0].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Vertical')[1].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Vertical')[2].text]) # Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Camera_Model_Parameters/CAHVOR_Model/Vector_Vertical
    self.data['o'] = np.array([self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Optical')[0].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Optical')[1].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Vector_Optical')[2].text]) # Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Camera_Model_Parameters/CAHVOR_Model/Vector_Optical
    self.data['r'] = np.array([self.lbl['label'][1][7][1][1][10][2].find('geom:Radial_Terms')[0].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Radial_Terms')[1].text,
                               self.lbl['label'][1][7][1][1][10][2].find('geom:Radial_Terms')[2].text]) # Product_Observational/Observation_Area/Discipline_Area/Geometry/Geometry_Lander[*]/Camera_Model_Parameters/CAHVOR_Model/Radial_Terms

    self.data['horizontal_fov'] = float(self.lbl['label'].find('Observation_Area')
                                                         .find('Discipline_Area')
                                                         .find('img:Imaging')
                                                         .find('img:Subframe')
                                                         .find('img:sample_fov').text)
    self.data['vertical_fov'] = float(self.lbl['label'].find('Observation_Area')
                                                       .find('Discipline_Area')
                                                       .find('img:Imaging')
                                                       .find('img:Subframe')
                                                       .find('img:line_fov').text)


if __name__=='__main__':

  gdal.UseExceptions()

  MISSION = MARS2020()

  ###################
  ### Set up QGIS ###
  ###################

  print("----------SETTING UP QGIS----------")

  QgsApplication.setPrefixPath("/usr", True)

  qgs = QgsApplication([], True)
  qgs.initQgis()

  project = QgsProject.instance()
  project.read(WORKSPACE_PATH + 'qgis/' + MISSION.PROJ_PATH) # if we want to load a project other than what is open

  ###################################
  ### Set up QGIS viewshed plugin ###
  ###################################
  
  Processing.initialize()

  # DEBUG: print all QGIS plugins in the registry (use this if you are having issues with paths and the processing plugins)
  # for alg in QgsApplication.processingRegistry().algorithms():
  #   print(alg.id(), "->", alg.displayName())

  visibility_provider = VisibilityProvider()
  visibility_provider.setActive(True)
  QgsApplication.processingRegistry().addProvider(visibility_provider)

  dtm_layer = project.mapLayersByName(MISSION.DTM_LAYER)[0]

  #########################################
  ### Iterate through all images in dir ###
  #########################################

  for mastcam_img_file in os.listdir(WORKSPACE_PATH + MISSION.IMGS_PATH):

    print("----------PROCESSING IMAGE " + mastcam_img_file + "----------")

    ############################
    ### Read data from label ###
    ############################

    if MISSION == MSL(): # PDS3
      mastcam_lbl = PDS3(WORKSPACE_PATH + MISSION.IMGS_PATH + mastcam_img_file, 
                        WORKSPACE_PATH + MISSION.LBLS_PATH + mastcam_img_file[:-4] + '.LBL')
    else: # PDS4
      mastcam_lbl = PDS4(WORKSPACE_PATH + MISSION.IMGS_PATH + mastcam_img_file, 
                        WORKSPACE_PATH + MISSION.LBLS_PATH + mastcam_img_file[:-4] + '.xml')

    #######################
    ### Create CSV file ###
    #######################

    x_coord = None
    y_coord = None
    rnav_rot = {}

    with open(WORKSPACE_PATH + 'qgis/' + MISSION.ROVER_PATH, 'r') as file:
      reader = csv.reader(file)
      for row in reader:
        if row[0] == 'frame': continue
        if int(row[1]) == mastcam_lbl.data['site'] and int(row[2]) == mastcam_lbl.data['drive']:
          x_coord = row[8]
          y_coord = row[7]
          rnav_rot['roll'] = math.radians(float(row[17])) # degrees
          rnav_rot['pitch'] = math.radians(float(row[18]))
          rnav_rot['yaw'] = math.radians(float(row[19]))
          break

    azimuth_1 = mastcam_lbl.data['fixed_instrument_azimuth'] - (mastcam_lbl.data['horizontal_fov'] / 2)
    azimuth_2 = mastcam_lbl.data['fixed_instrument_azimuth'] + (mastcam_lbl.data['horizontal_fov'] / 2)
    angle_up = mastcam_lbl.data['fixed_instrument_elevation'] + (mastcam_lbl.data['vertical_fov'] / 2)
    angle_down = mastcam_lbl.data['fixed_instrument_elevation'] - (mastcam_lbl.data['vertical_fov'] / 2)

    if azimuth_1 < 0: azimuth_1 += 360

    with open(WORKSPACE_PATH + 'qgis/csv_files/' + 'csv_' + mastcam_lbl.img_id + '.csv', 'w') as file:
      writer = csv.writer(file)
      writer.writerow(['X', 'Y', 'OFFSET', 'AZIMUTH1', 'AZIMUTH2', 'ANGLEUP', 'ANGLEDOWN'])
      writer.writerow([x_coord, y_coord, 1.97, azimuth_1, azimuth_2, angle_up, angle_down])

    csv_file = 'file://' + WORKSPACE_PATH + 'qgis/csv_files/' + 'csv_' + mastcam_lbl.img_id + '.csv' + '?delimiter=,&crs=USER:100000&xField=X&yField=Y'

    # Create group for easier viewing on QGIS
    root = QgsProject.instance().layerTreeRoot()
    group = root.insertGroup(0, mastcam_img_file[:-4])

    # Import CSV
    csv_layer = QgsVectorLayer(csv_file, 'csv', 'delimitedtext')
    project.addMapLayer(csv_layer, False)
    group.insertChildNode(0, QgsLayerTreeLayer(csv_layer))

    #################################
    ### Create viewpoint from CSV ###
    #################################
    
    params = {'OBSERVER_ID': '',
              'OBSERVER_POINTS': csv_layer, 
              'DEM': dtm_layer,
              'RADIUS': 5000, # default
              'RADIUS_FIELD': '',
              'RADIUS_IN_FIELD': '',
              'OBS_HEIGHT': next(csv_layer.getFeatures())['OFFSET'],
              'OBS_HEIGHT_FIELD': 'OFFSET',
              'TARGET_HEIGHT': 0,
              'TARGET_HEIGHT_FIELD': '',
              'AZIM_1_FIELD': 'AZIMUTH1',
              'AZIM_2_FIELD': 'AZIMUTH2',
              'ANGLE_UP_FIELD': 'ANGLEUP',
              'ANGLE_DOWN_FIELD': 'ANGLEDOWN',
              'OUTPUT': WORKSPACE_PATH + 'qgis/viewpoints/' + 'viewpoint_' + mastcam_lbl.img_id}
    
    out = processing.run('visibility:createviewpoints', params)

    viewpoint_layer = QgsVectorLayer(out['OUTPUT'], 'viewpoint_'+mastcam_lbl.img_id, 'ogr')
    project.addMapLayer(viewpoint_layer, False)
    group.insertChildNode(0, QgsLayerTreeLayer(viewpoint_layer))

    ####################################
    ### Create viewshed at viewpoint ###
    ####################################

    # DEBUG: For debugging the processing tool--
    # print(processing.algorithmHelp('visibility:viewshed'))

    params = {'ANALYSIS_TYPE': 0, # Binary viewshed
              'OBSERVER_POINTS': viewpoint_layer,
              'DEM': dtm_layer,
              'USE_CURVATURE': False,
              'REFRACTION': 0.13,
              'OPERATOR': 0, # Addition
              'OUTPUT': WORKSPACE_PATH + 'qgis/viewsheds/' + 'viewshed_' + mastcam_lbl.img_id}
    
    out = processing.run('visibility:viewshed', params)

    viewshed_layer = QgsRasterLayer(out['OUTPUT'], 'viewshed_' + mastcam_lbl.img_id)
    project.addMapLayer(viewshed_layer, False)
    group.insertChildNode(0, QgsLayerTreeLayer(viewshed_layer))

    # color the visible area
    fcn = QgsColorRampShader()
    fcn.setColorRampType(QgsColorRampShader.Discrete)

    pcolor = [QgsColorRampShader.ColorRampItem(0, QColor('r=132, g=0, b=0, a=255'), '0'), QgsColorRampShader.ColorRampItem(1, QColor('r=132, g=0, b=0, a=50'), '1')]

    # renderer = QgsPalettedRasterRenderer(viewpoint_layer.dataProvider(), 1, QgsPalettedRasterRenderer.colorTableToClassData(pcolor))
    renderer = QgsPalettedRasterRenderer(viewshed_layer.dataProvider(), 1, [QgsPalettedRasterRenderer.Class(0, QColor(132, 0, 0, 0), '0'), 
                                                                             QgsPalettedRasterRenderer.Class(1, QColor(132, 0, 0, 255), '1')])
    viewshed_layer.setRenderer(renderer)
    viewshed_layer.triggerRepaint()

    group.setExpanded(True)
    group.setExpanded(False)

    # Save to file
    project.write(WORKSPACE_PATH + 'qgis/' + MISSION.PROJ_PATH)

    # Clean up GDAL and QGIS
    viewshed_layer_gdal = None 
  
  sys.stdout.flush()
  qgs.exitQgis()
