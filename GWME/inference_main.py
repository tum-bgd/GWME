# =======================================================================================================================================================
# inference_main.py
# Author: Jiapan Wang
# Created Date: 26/05/2023
# Description: inference test images with the trained object detection model, save detection images and detection bbox as geojson format.
# =======================================================================================================================================================
"""
Args:
    image_path: Path to images folder.
    model_path: Path to saved model.
    label_path: Path to label map.
    output_dir: Path to inference results.
"""

from absl import app
from absl import flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import json
import math
import geojson
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

FLAGS = flags.FLAGS

flags.DEFINE_string("image_path", "images/test/", "Path to images folder.")
flags.DEFINE_string("model_path", None, "Path to saved model.")
flags.DEFINE_string("label_path", None, "Path to label map.")
flags.DEFINR_string("output_dir", "./prediction/", "Path to results.")

flags.mark_flag_as_required('image_path')
flags.mark_flag_as_required('model_path')
flags.mark_flag_as_required('label_path')
flags.mark_flag_as_required('output_path')

# Load test images
def load_images(path):
#     base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    
    # Get the list of all files and directories
    dir_list = os.listdir(path)
    # print("Files and directories in '", path, "' :")
    # prints all files
    print(dir_list)
    
    filenames = dir_list
    image_paths = []
    for filename in filenames:
#         image_path = tf.keras.utils.get_file(fname=filename,
#                                             origin=path + filename,
#                                             untar=False)
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

def detection_to_dictionary(task_id, boxes, classes, scores):
    
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
    
    output_dir = FLAGS.output_dir + "/json"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    jsonfile = task_id + ".geojson"
    out_path = os.path.join(output_dir, jsonfile)
       
    save_dict_to_geojson(pred_dict, out_path)

def save_dict_to_geojson(dictionary, out_path): 

    with open(out_path, "w", encoding='utf-8') as outfile:
        json.dump(dictionary, outfile)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def merge_all_geojson_to_one(input_dir):
    
    file_list = os.listdir(input_dir)
    print(file_list)
   
    merge = list()
    
    for file in file_list:
        input_path = os.path.join(input_dir, file)
        print(input_path)
        with open(input_path, 'r') as input_file:
            merge.extend(geojson.load(input_file)["features"])
#         print(merge)

    geo_collection = geojson.FeatureCollection(merge)

    return geo_collection

def main(argv):
    image_Paths = load_images(FLAGS.image_path)
    
    # Load saved model and build the detection function
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(FLAGS.model_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # Load label map
    category_index = label_map_util.create_category_index_from_labelmap(FLAGS.label_path,
                                                                        use_display_name=True)

    output_preview_path = FLAGS.output_dir + "/preview/"
    if not os.path.exists(output_preview_path):
        os.makedirs(output_preview_path)

    for path in image_Paths:

        print('Running inference for {}... '.format(path), end='')

        image_np = load_image_into_numpy_array(path)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)

        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)
        
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    #     print("detections",detections)   
        
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
        image_np_with_detections = image_np.copy()
        
        # Non max suppression
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            detections['detection_boxes'], 
            detections['detection_scores'], 
            detections['detection_boxes'].shape[0], 
            iou_threshold=1.0,
            score_threshold=0.6,
            soft_nms_sigma=0.5)
        
        selected_boxes = tf.gather(detections['detection_boxes'], selected_indices)
        selected_classes = tf.gather(detections['detection_classes'], selected_indices)
        
        selected_boxes_np = selected_boxes.numpy()
        selected_classes_np = selected_classes.numpy()
        selected_scores_np = selected_scores.numpy()
        selected_num_detections = len(selected_indices)
            
        vis_image = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            selected_boxes_np,
            selected_classes_np,
            selected_scores_np,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)
        
        # image id
        task_id = os.path.splitext(os.path.basename(path))[0]
        
        # save prediction image
        output_path = output_preview_path  + task_id +".png"
        viz_utils.save_image_array_as_png(vis_image, output_path)
    
        # save predition to json
        bboxes = (selected_boxes_np*256).astype(np.int)    
        detection_to_dictionary(task_id, bboxes, selected_classes_np, selected_scores_np)
        
        # plt.figure()
        # plt.imshow(image_np_with_detections)
        print('Done')
    # plt.show()

    input_dir = FLAGS.output_dir + "/json/"
    geojson_all = merge_all_geojson_to_one(input_dir)

    output_path = FLAGS.output_dir + "merged_prediction.geojson"
    with open(output_path, 'w', encoding='utf-8') as output_file:
        geojson.dump(geojson_all, output_file)

    print("done")

if __name__ == '__main__':
    app.run(main)