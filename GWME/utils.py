# =======================================================================================================================================================
# utils.py
# Author: Jiapan Wang
# E-mail: jiapan.wang@tum.de
# Created Date: 01/06/2023
# Description: Calculate the weight of distance, image correlation and attention between the reference area and the target area.
# =======================================================================================================================================================
"""Utility functions."""

import os
import pathlib
import math
import numpy as np
from PIL import Image
import json
import geojson


def load_images(path):
    '''Load images'''
    # base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    # Get the list of all files and directories
    filenames = os.listdir(path)
    image_paths = []

    for filename in filenames:
        image_path = pathlib.Path(path+filename)
        image_paths.append(str(image_path))
    
    return image_paths


def pixel_coords_zoom_to_lat_lon(PixelX, PixelY, zoom):
    MapSize = 256 * math.pow(2, zoom)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon = 360 * x
    lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi

    return lon, lat


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)"""
    return np.array(Image.open(path))


def parse_tile_name(name):
    zoom, TileX, TileY = [int(x) for x in name.split(".")]

    return TileX, TileY, zoom


def pixel_coords_to_latlon(task_id, bbox_polygon):
    TileX, TileY, zoom = parse_tile_name(task_id)
    PixelX = TileX * 256
    PixelY = TileY * 256
    coords = bbox_polygon
    # print(TileX, TileY, zoom)

    translated = [[PixelX + y, PixelY + x] for x, y in coords]

    return [list(pixel_coords_zoom_to_lat_lon(x, y, zoom)) for x, y in translated]


def save_dict_to_geojson(dictionary, out_path): 
    with open(out_path, "w", encoding='utf-8') as outfile:
        json.dump(dictionary, outfile)


def merge_all_geojson_to_one(input_dir):
    file_list = os.listdir(input_dir)

    merge = []

    print("start merging geojson ...")
    for file in file_list:
        input_path = os.path.join(input_dir, file)
        print(input_path)
        with open(input_path, 'r') as input_file:
            merge.extend(geojson.load(input_file)["features"])

    return geojson.FeatureCollection(merge)
