# =======================================================================================================================================================
# evaluation_geojson.py
# Author: Jiapan Wang
# E-mail: jiapan.wang@tum.de
# Created Date: 05/06/2023
# Description: Evaluating predicted geojson objects and reference geojson objects.
# =======================================================================================================================================================
"""
Args:
    prediction_path: Path to prediction geojson path.
    reference_path: Path to reference geojson path.
    output_path: Path to evaluated geojson objects.

Usage Case:
    python evaluation_geojson.py \
            --prediction_path="./case_study/cameroon/predictions/prediction-ensemble/merged_prediction_multi_heads_attention.geojson" \
            --reference_path="./case_study/cameroon/reference/building_building_.geojson" \
            --output_path="./case_study/cameroon/evaluation/"
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import geojson
import time
import pathlib
from tqdm import tqdm
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("prediction_path", None, "Path to prediction geojson path.")
flags.DEFINE_string("reference_path", None, "Path to reference geojson path.")
flags.DEFINE_string("output_path", None, "Path to evaluated geojson objects.")

flags.mark_flag_as_required('prediction_path')
flags.mark_flag_as_required('reference_path')
flags.mark_flag_as_required('output_path')


def load_path(dir_path):
    """List paths of all files under dir_path"""
    file_list = os.listdir(dir_path)

    file_paths = []

    for filename in file_list:
        file_path = pathlib.Path(dir_path+filename)
        file_paths.append(str(file_path))

    return file_paths


def load_geojson(path):
    """Load geojson features as dictionary from geojson file path"""
    with open(path, "r") as input_file:
        geojson_dict = geojson.load(input_file)

    return geojson_dict


def prediction_filter_by_score(pred_dict, min_score):
    """Filter predictions by score threshold"""
    new_pred_dict = {'type': pred_dict['type'], 'features': []}
    for feature in pred_dict['features']:
        if feature['properties']['score'] >= min_score:
            new_pred_dict['features'].append(feature)

    return new_pred_dict


def prediction_to_gdf(dict):
    """Convert prediction geojson dict to geodataframe"""
    geo_series_list = []
    # filter predictions by extend
    (left, bottom, right, top) = (10.1926319999999997,5.6555059999999999, 10.2352519999999991,5.6722060000000001)
    ref_extend = Polygon([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)])
    id = 0
    for feature in dict['features']:
        pred_polygon = Polygon(feature['geometry']['coordinates'][0])
        if (ref_extend.contains(pred_polygon)):
            geo_series = {
                'task_id': feature['properties']['task_id'],
                'score': feature['properties']['score'],
                'prediction_id': id,
                'type': feature['geometry']['type'],
                'if_correct': False,
                'geometry': pred_polygon
            }
            geo_series_list.append(geo_series)
            id += 1

    return gpd.GeoDataFrame(geo_series_list)


def reference_to_gdf(dict):
    """Convert reference geojson dict to geodataframe"""
    geo_series_list = []
    for i, feature in enumerate(dict['features']):
        geo_series = {
            'ref_id': i,
            'type': feature['geometry']['type'],
            'if_detected': False,
            'geometry': get_bbox(Polygon(feature['geometry']['coordinates'][0]))
        }
        geo_series_list.append(geo_series)

    return gpd.GeoDataFrame(geo_series_list)


def get_bbox(geometry):
    """Get bbox of geometry"""
    xmin, ymin, xmax, ymax = geometry.bounds

    return Polygon.from_bounds(xmin, ymin, xmax, ymax)


def calculate_IOU(box1, box2):
    """Calculate IOU between two boxes"""
    intersect_df = gpd.overlay(box1, box2, how="intersection")
    
    union_df = gpd.overlay(box1, box2, how="union")
    union_df = union_df.dissolve()
    
    iou = intersect_df.area / union_df.area
    
    return iou.iloc[0]


def main_eval(prediction_path, reference_path, FILTER_THRESHOLD, IOU_THRESHOLD, COVERAGE_THRESHOLD):
    '''Evaluation for entire geojson'''
    all_predictions_gdf = gpd.GeoDataFrame()
    all_ref_gdf = gpd.GeoDataFrame()

    pred_geojson = load_geojson(prediction_path)
    ref_geojson = load_geojson(reference_path)

    filtered_pred_geojson = prediction_filter_by_score(pred_geojson, FILTER_THRESHOLD)

    pred_geodf = prediction_to_gdf(filtered_pred_geojson)
    ref_geodf = reference_to_gdf(ref_geojson)

    intersection_df = gpd.overlay(pred_geodf, ref_geodf, how='intersection')

    TP = 0
    for intersect in tqdm(intersection_df.itertuples()):
        pred_id = intersect.prediction_id
        ref_id = intersect.ref_id

        pred_item = pred_geodf.loc[pred_geodf['prediction_id'] == pred_id]
        ref_item = ref_geodf.loc[ref_geodf['ref_id'] == ref_id]
        iou = calculate_IOU(pred_item, ref_item)

        if iou >= IOU_THRESHOLD:
            pred_geodf.loc[pred_geodf['prediction_id'] == pred_id, 'if_correct'] = True #====> TP
            ref_geodf.loc[ref_geodf['ref_id'] == ref_id,'if_detected'] = True #====> TP
            TP += 1

        #  no TP, but prediction mostly covered by the ref building
        elif (intersect.geometry.area/pred_item.geometry.area).iloc[0] >= COVERAGE_THRESHOLD:
            pred_geodf.loc[pred_geodf['prediction_id'] == pred_id, 'if_correct'] = True #====> TP
            ref_geodf.loc[ref_geodf['ref_id'] == ref_id,'if_detected'] = True #====> TP
            TP += 1

    total_pred = pred_geodf.shape[0]

    TP = pred_geodf.loc[pred_geodf['if_correct'] == True].shape[0]
    FP = total_pred - TP
    FN = ref_geodf.loc[ref_geodf['if_detected'] == False].shape[0]

    precision = TP/total_pred
    recall = TP/(TP+FN)
    f1 = 2*(precision*recall/(precision+recall))
    accuracy = TP/(TP+FP+FN)
    # mcc = (tn * TP - FN * FP) / math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(tn + FN))
    print(f"metrics: TP = {TP}, FP = {FP}, FN = {FN}, total = {total_pred}, \nprecision = {precision}, recall = {recall}, f1 = {f1}, accuracy = {accuracy}")

    result_metrics = {
        "total_pred": total_pred,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

    all_predictions_gdf = pd.concat([all_predictions_gdf, pred_geodf])
    all_ref_gdf = pd.concat([all_ref_gdf, ref_geodf])


    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    all_predictions_gdf.to_file(FLAGS.output_path + "all_predictions_eval.geojson", driver="GeoJSON")
    all_ref_gdf.to_file(FLAGS.output_path + "all_reference_eval.geojson", driver="GeoJSON")    

    return result_metrics


def main(argv):
    prediction_path = FLAGS.prediction_path
    reference_path = FLAGS.reference_path

    print("start evaluating...")
    start_time = time.time()

    # threshold
    COVERAGE_THRESHOLD = 0.5 # if the ratio of intersection area to prediction area higher than this threshold, it means the prediction is mostly covered by reference
    IOU_THRESHOLD = 0.5
    FILTER_THRESHOLD = 0.1

    metrics = main_eval(prediction_path, reference_path, FILTER_THRESHOLD, IOU_THRESHOLD, COVERAGE_THRESHOLD)

    print(f"evaluation metrics:\n{metrics}")    

    print(f"time used: {time.time() - start_time}")


if __name__ == '__main__':
    app.run(main)
