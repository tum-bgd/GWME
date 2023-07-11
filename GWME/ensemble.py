# =======================================================================================================================================================
# ensemble.py
# Author: Jiapan Wang
# E-mail: jiapan.wang@tum.de
# Created Date: 02/06/2023
# Description: Ensemble predications generated from different object detection models.
# =======================================================================================================================================================
"""
Args:
    prediction_path: Path to predictions.
    weights_path: Path to weights json file.
    mode: Weighting mode (average|similarity|distance|attention).

Usage Case:
    python ensemble.py \
            --prediction_path="./case_study/cameroon/predictions/" \
            --weights_path="./case_study/cameroon/weights/all_weights.json"\
            --mode="average"
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    
import warnings
import pathlib
import numpy as np
import json
import math
from tqdm import tqdm
import geojson
from ensemble_boxes import *
from prediction_to_geojson import detection_to_geojson, merge_all_geojson_to_one


from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("prediction_path", None, "Path to predictions.")
flags.DEFINE_string("weights_path", None, "Path to weights json file.")
flags.DEFINE_string("mode", "average", "Weighting mode (average|similarity|distance|attention).")

flags.mark_flag_as_required('prediction_path')
flags.mark_flag_as_required('weights_path')
flags.mark_flag_as_required('mode')

IMAGE_SIZE = 256

def normalize_box(box):

    (x1, y1, x2, y2) = box[0]/IMAGE_SIZE, box[1]/IMAGE_SIZE, box[2]/IMAGE_SIZE, box[3]/IMAGE_SIZE
    norm_box = [x1, y1, x2, y2]

    return norm_box

def load_weights():

    with open(FLAGS.weights_path,"r") as file:
        weights_dict_list = json.load(file)

    return weights_dict_list

def ensemble(weights_type):
    pred_dir = os.listdir(FLAGS.prediction_path)
    # ensemble target directory
    ensemble_dir = pred_dir.pop(pred_dir.index("prediction-ensemble"))

    print("ensemble dir: ", ensemble_dir)
    print("prediction dir: ", pred_dir)

    json_path = FLAGS.prediction_path + pred_dir[0] + "/json"
    json_list = os.listdir(json_path) # geojson filename list

    weights_dict_list = load_weights()
    
    # ensemble for each tile
    for i, json_filename in enumerate(tqdm(json_list)):

        # bbox, score, label combined from multiple models
        boxes_list = []
        score_list = []
        label_list = []      

        task_id = os.path.splitext(os.path.basename(json_filename))[0]

        for pred_model in pred_dir:

            # bbox, score, label from single model
            bbox = []
            score = []
            label = []

            # predictions/model/json/task_id.geojson
            input_path = FLAGS.prediction_path + pred_model + "/json/" + json_filename 

            with open(input_path, 'r') as input_file:
                geojson_dict = geojson.load(input_file)
            geojson_dict['features']

            for feature in geojson_dict['features']:
                label.append(feature['properties']['prediction_class'])
                score.append(feature['properties']['score'])
                bbox.append(normalize_box(feature['properties']['bbox']))

            boxes_list.append(bbox)
            score_list.append(score)
            label_list.append(label)

        iou_thr = 0.5
        skip_box_thr = 0.0001
        conf_type = "box_and_model_avg"

        weight_dict = weights_dict_list[i]

        if weights_type == "average":
            weights = [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8]
        elif weights_type == "distance":
            weights = weight_dict["inverse_distance_weights"]
        elif weights_type == "similarity":
            weights = weight_dict["image_similarity_weights"]
        elif weights_type == "attention":
            weights = weight_dict["attention_map_weights"]
        else:
            warnings.warn('Please select an appropriate weight type: "average" or "distance" or "similarity"')
            exit()

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, score_list, label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
        boxes = (np.array(boxes)*256).astype(np.int)  

        output_path = FLAGS.prediction_path + "prediction-ensemble/json/"
        detection_to_geojson(task_id, boxes, labels, scores, output_path)

        # break

    input_dir = FLAGS.prediction_path + "prediction-ensemble/json/"
    geojson_all = merge_all_geojson_to_one(input_dir)

    output_path = FLAGS.prediction_path + "prediction-ensemble/merged_prediction.geojson"
    with open(output_path, 'w', encoding='utf-8') as output_file:
        geojson.dump(geojson_all, output_file)

    print("done")

    return

def main(argv):
    print("hello, ensemble")
    ensemble(FLAGS.mode)

if __name__ == "__main__":
    app.run(main)

    
