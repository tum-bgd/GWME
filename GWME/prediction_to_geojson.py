# =======================================================================================================================================================
# prediction_to_geojson.py
# Author: Jiapan Wang
# E-mail: jiapan.wang@tum.de
# Created Date: 02/06/2023
# Description: Convert prediction from image to geojson.
# =======================================================================================================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import pathlib
import numpy as np
import json
import math
import geojson

def pixel_coords_zoom_to_lat_lon(PixelX, PixelY, zoom):
    MapSize = 256 * math.pow(2, zoom)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon = 360 * x
    lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi

    return lon, lat

def parse_tile_name(name):
    zoom, TileX, TileY = [int(x) for x in name.split(".")]
    return TileX, TileY, zoom

def pixel_coords_to_latlon(task_id, bbox_polygon):
    TileX, TileY, zoom = parse_tile_name(task_id)
    PixelX = TileX * 256
    PixelY = TileY * 256
    coords = bbox_polygon
#     print(TileX, TileY, zoom)
    
    translated = [[PixelX + y, PixelY + x] for x, y in coords]
    transformed = [[i for i in pixel_coords_zoom_to_lat_lon(x, y, zoom)] for x, y in translated]

    return transformed

def detection_to_geojson(task_id, boxes, classes, scores, output_path):
    
    pred_dict = {
        "type": "FeatureCollection",      
        "features":[]
    }
        
#     print("pred_dict", pred_dict)
    
    for i, bbox in enumerate(boxes):

        bbox = [max(0, min(255, int(x))) for x in bbox[:4]]
        (left, right, top, bottom) = (int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3]))
        bbox_polygon=[(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]
        
        bbox_coords = pixel_coords_to_latlon(task_id, bbox_polygon)
                
        new_pred = {
            "type": "Feature",
            "properties": {
                "task_id": task_id,
                "prediction_id": i,
                "prediction_class": int(classes[i]),
                "score": float(scores[i]),
                "bbox": bbox,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [bbox_coords]
            }
        }
        
#         print(i, new_pred)
        pred_dict["features"].append(new_pred)
        
#     print("pred_dict", pred_dict)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    jsonfile = task_id + ".geojson"
    out_path = os.path.join(output_path, jsonfile)
       
    save_dict_to_geojson(pred_dict, out_path)

def save_dict_to_geojson(dictionary, out_path): 

    with open(out_path, "w", encoding='utf-8') as outfile:
        json.dump(dictionary, outfile)

def merge_all_geojson_to_one(input_dir):
    
    file_list = os.listdir(input_dir)
    # print(file_list)
   
    merge = list()
    print("start merging geojson ...")
    for file in file_list:
        input_path = os.path.join(input_dir, file)
        print(input_path)
        with open(input_path, 'r') as input_file:
            merge.extend(geojson.load(input_file)["features"])
#         print(merge)

    geo_collection = geojson.FeatureCollection(merge)

    return geo_collection