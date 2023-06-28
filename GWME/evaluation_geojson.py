# =======================================================================================================================================================
# evaluation_geojson.py
# Author: Jiapan Wang
# Created Date: 05/06/2023
# Description: Evaluating predicted geojson objects and reference geojson objects.
# =======================================================================================================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import numpy as np
import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import geojson
import time
import pathlib

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("prediction_path", None, "Path to prediction geojson path.")
flags.DEFINE_string("reference_path", None, "Path to reference geojson path.")

flags.mark_flag_as_required('prediction_path')
flags.mark_flag_as_required('reference_path')

def load_path(dir_path):
    """list paths of all files under dir_path
    """

    file_list = os.listdir(dir_path)

    file_paths = []

    for filename in file_list:
        file_path = pathlib.Path(dir_path+filename)
        file_paths.append(str(file_path))

    return file_paths

def load_geojson(path):
    """load geojson features as dictionary from geojson file path
    """

    with open(path, "r") as input_file:
        geojson_dict = geojson.load(input_file)

    return geojson_dict

def prediction_filter_by_score(pred_dict, min_score):
    """filter predictions by score threshold
    """
    new_pred_dict = {}
    new_pred_dict['type'] = pred_dict['type']
    new_pred_dict['features'] = []
    for feature in pred_dict['features']:
        if feature['properties']['score'] >= min_score:
            new_pred_dict['features'].append(feature)

    return new_pred_dict

def prediction_to_gdf(dict):
    """convert prediction geojson dict to geodataframe
    """
    geo_series_list = []
    # filter predictions by extend
    (left, bottom, right, top) = (10.1926319999999997,5.6555059999999999, 10.2352519999999991,5.6722060000000001)
    ref_extend = Polygon([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]) 
    id = 0
    for feature in dict['features']:
        pred_polygon = Polygon(feature['geometry']['coordinates'][0])
        if(ref_extend.contains(pred_polygon)):
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
        else:
            continue
    geo_df = gpd.GeoDataFrame(geo_series_list)

    return geo_df

def reference_to_gdf(dict):
    """convert reference geojson dict to geodataframe
    """
    geo_series_list = []
    for i, feature in enumerate(dict['features']):
        geo_series = {
            'ref_id': i,
#             'label': feature['properties']['label'],
            'type': feature['geometry']['type'],
            'if_detected': False,
            'geometry': get_bbox(Polygon(feature['geometry']['coordinates'][0]))
        }
        geo_series_list.append(geo_series)
    geo_df = gpd.GeoDataFrame(geo_series_list)
    
    return geo_df

def get_bbox(geometry):
    """ get bbox of geometry
    """
    xmin, ymin, xmax, ymax = geometry.bounds
    bbox_polygon = Polygon.from_bounds(xmin, ymin, xmax, ymax)
    
    return bbox_polygon

# def intersection_gdf(gdf1, gdf2):
#     """find intersected geometry between two geo dataframe
#     """
    
# #     intersection = gdf1.geometry.overlaps(gdf2.geometry, align=True)
#     intersection_gdf = gpd.overlay(gdf1, gdf2, how='intersection')
#     print("intersection: \n", intersection_gdf['properties_1'])
    
#     gdf1.plot()
#     gdf2.plot()
#     intersection_gdf.plot()
    
#     return intersection_gdf

def calculate_IOU(box1, box2):
    """Calculate IOU between two boxes
    """
    intersect_df = gpd.overlay(box1, box2, how="intersection")
    
    union_df = gpd.overlay(box1, box2, how="union")
    union_df = union_df.dissolve()
    
    iou = intersect_df.area / union_df.area
    
    return iou.iloc[0]


# # calculate metrics precision, recall, accuracy, f1, MCC
# def metrics_statistic(results):

#     # print(len(results))
#     tp = 0
#     fp = 0
#     fn = 0
#     tn = 0
#     total_pred = 0
#     for tile_result in results:
#         # print("tile id", tile_result['tile_id'])
#         tp += tile_result['TP']
#         fp += tile_result['FP']
#         fn += tile_result['FN']
#         total_pred += tile_result['total_pred']
        
#     precision = tp/total_pred 
#     recall = tp/(tp+fn)
#     f1 = 2*(precision*recall/(precision+recall))
#     mcc = (tn * tp - fn * fp) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
#     accuracy = tp/(tp+fp+fn)
#     print(f"metrics: TP = {tp}, FP = {fp}, FN = {fn}, total = {total_pred}, \nprecision = {precision}, recall = {recall}, f1 = {f1}, accuracy = {accuracy}")
#     metrics = {
#         "TP": tp,
#         "FP": fp,
#         "FN": fn,
#         "total_pred": total_pred,
#         "precision": precision,
#         "accuracy": accuracy,
#         "recall": recall,
#         "f1": f1
#     }

#     return metrics

# tile-based evaluation
# def main_eval_tile(prediction_dir, reference_dir, FILTER_THRESHOLD, IOU_THRESHOLD, COVERAGE_THRESHOLD):

#     # data dir
#     # # attention
#     # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_attention/"
#     # # distance
#     # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_distance/"
#     # similarity
#     # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_similarity/"
#     # # average
#     # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_average/"
#     # single model
#     # prediction_dir = "./ensemble/predictions/prediction-model-07/json/"
#     # reference_dir = "./evaluation/reference/tile/"

#     pred_geojson_paths = load_path(prediction_dir)
#     ref_geojson_paths = load_path(reference_dir)

#     results = []

#     for i, pred_path in enumerate(pred_geojson_paths):
        
#         tile_id = os.path.splitext(os.path.basename(pred_path))[0]
#         # print(f"{i} evaluation for ......", tile_id)
#     #     i = 50
#         pred_geojson = load_geojson(pred_geojson_paths[i])
#         ref_geojson = load_geojson(ref_geojson_paths[i])

#         filtered_pred_geojson = prediction_filter_by_score(pred_geojson, FILTER_THRESHOLD)

#         # print("pred geojson:", len(filtered_pred_geojson['features']), filtered_pred_geojson)
#         # print("ref geojson:", len(ref_geojson['features']), ref_geojson)

#         pred_geodf = prediction_to_gdf(filtered_pred_geojson)
#         ref_geodf = reference_to_gdf(ref_geojson)

#     #     print("pred geodf: \n", pred_geodf)
#     #     print("ref geodf: \n", ref_geodf)
#         TP = 0
        
#         # intersction gdf
#         if pred_geodf.shape[0] != 0 and ref_geodf.shape[0]!= 0:
#             intersection_df = gpd.overlay(pred_geodf, ref_geodf, how='intersection')
#             # print("intersection: \n", intersection_df)
            
#             # print("intersection from gdf1: \n", pred_geodf.loc[intersection_df['prediction_id']])
#             # print("intersection from gdf2: \n", ref_geodf.loc[intersection_df['ref_id']])

#             # print(type(pred_geodf.loc[intersect_df['prediction_id']]))

#             # ax = ref_geodf.plot(linewidth = 2, color="white", edgecolor="green")
#             # ax2 = pred_geodf.plot(ax=ax, color="red", linewidth = 1, alpha=0.5, edgecolor="red")
#             # intersection_df.plot(ax = ax2, linewidth = 1, color="white", edgecolor="black")    

#             for i, intersect in enumerate(intersection_df.itertuples()):
#                 # print(f"intersect {i}======================================================================\n")
#                 pred_id = intersect.prediction_id
#                 ref_id = intersect.ref_id

#                 pred_item = pred_geodf.loc[pred_geodf['prediction_id'] == pred_id]
#                 ref_item = ref_geodf.loc[ref_geodf['ref_id'] == ref_id]
#                 iou = calculate_IOU(pred_item, ref_item)

#     #             print("iou:", iou)

#             #     print("pred contain ref",intersect.geometry.area/pred_item.geometry.area)
#             #     print("ref contain pred",intersect.geometry.area/ref_item.geometry.area)


#                 if iou >= IOU_THRESHOLD:
#                     pred_geodf.loc[pred_geodf['prediction_id'] == pred_id, 'if_correct'] = True #====> TP
#                     ref_geodf.loc[ref_geodf['ref_id'] == ref_id,'if_detected'] = True #====> TP
#                     TP += 1
#                 #  no TP, but prediction mostly covered by the ref building
#                 elif (intersect.geometry.area/pred_item.geometry.area).iloc[0] >= COVERAGE_THRESHOLD:
#                     pred_geodf.loc[pred_geodf['prediction_id'] == pred_id, 'if_correct'] = True #====> TP
#                     ref_geodf.loc[ref_geodf['ref_id'] == ref_id,'if_detected'] = True #====> TP
#                     TP += 1

#         total_pred = pred_geodf.shape[0]
#         total_ref = ref_geodf.shape[0]

#         FP = total_pred - TP
#         FN = ref_geodf.loc[ref_geodf['if_detected'] == False].shape[0]

#     #     print("pred geodf: \n", pred_geodf)
#     #     print("ref geodf: \n", ref_geodf)   
#         # print(f"metrics: TP = {TP}, FP = {FP}, FN = {FN}\n")


#         tile_result = {
#             "tile_id": tile_id,
#             "total_pred": total_pred,
#             "TP": TP,
#             "FP": FP,
#             "FN": FN
#         }
        
#         results.append(tile_result)

#     return results


# evaluation for entire geojson
def main_eval(prediction_path, reference_path, FILTER_THRESHOLD, IOU_THRESHOLD, COVERAGE_THRESHOLD):

    all_predictions_gdf = gpd.GeoDataFrame()
    all_ref_gdf = gpd.GeoDataFrame()

    pred_geojson = load_geojson(prediction_path)
    ref_geojson = load_geojson(reference_path)

    filtered_pred_geojson = prediction_filter_by_score(pred_geojson, FILTER_THRESHOLD)

    pred_geodf = prediction_to_gdf(filtered_pred_geojson)
    ref_geodf = reference_to_gdf(ref_geojson)

    # print("pred_geodf", pred_geodf)
    # print("ref_geodf", ref_geodf)

    TP = 0

    intersection_df = gpd.overlay(pred_geodf, ref_geodf, how='intersection')
    # print("intersection: \n", intersection_df)

    for i, intersect in enumerate(intersection_df.itertuples()):
        pred_id = intersect.prediction_id
        ref_id = intersect.ref_id

        pred_item = pred_geodf.loc[pred_geodf['prediction_id'] == pred_id]
        ref_item = ref_geodf.loc[ref_geodf['ref_id'] == ref_id]
        iou = calculate_IOU(pred_item, ref_item)

    #             print("iou:", iou)
        # print(i, intersect)

    #     print("pred contain ref",intersect.geometry.area/pred_item.geometry.area)
    #     print("ref contain pred",intersect.geometry.area/ref_item.geometry.area)

        if iou >= IOU_THRESHOLD:
            pred_geodf.loc[pred_geodf['prediction_id'] == pred_id, 'if_correct'] = True #====> TP
            ref_geodf.loc[ref_geodf['ref_id'] == ref_id,'if_detected'] = True #====> TP
            TP += 1
        #  no TP, but prediction mostly covered by the ref building
        elif (intersect.geometry.area/pred_item.geometry.area).iloc[0] >= COVERAGE_THRESHOLD:
            pred_geodf.loc[pred_geodf['prediction_id'] == pred_id, 'if_correct'] = True #====> TP
            ref_geodf.loc[ref_geodf['ref_id'] == ref_id,'if_detected'] = True #====> TP
            TP += 1
        
        # ax = ref_item.plot(linewidth = 2, color="white", edgecolor="green")
        # ax2 = pred_item.plot(ax=ax, color="red", linewidth = 1, alpha=0.5, edgecolor="red")
        
        # print(pred_geodf.loc[pred_geodf['prediction_id'] == pred_id, 'if_correct'])
        # print(ref_geodf.loc[ref_geodf['ref_id'] == ref_id,'if_detected'])
    #     break

    total_pred = pred_geodf.shape[0]
    total_ref = ref_geodf.shape[0]
    
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

    all_predictions_gdf.to_file("./evaluation/all_predictions_eval_base.geojson", driver="GeoJSON")  
    all_ref_gdf.to_file("./evaluation/all_reference_eval_base.geojson", driver="GeoJSON")    

    return result_metrics


if __name__ == "__main__":

    import csv
    # data dir
    # prediction_dir = "./ensemble/predictions/prediction-base-model/json/"
    # # attention
    # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_multi_heads_attention/"
    # # distance
    # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_distance/"
    # similarity
    # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_similarity/"
    # # average
    # prediction_dir = "./ensemble/predictions/prediction-ensemble/json_average_new/"
    # single model
    # prediction_dir = "./ensemble/predictions/prediction-model-09/json/"
    # reference_dir = "./evaluation/reference/tile/"


    # prediction_path = "./ensemble/predictions/prediction-ensemble/merged_prediction_multi_heads_attention.geojson"
    # prediction_path = "./ensemble/predictions/prediction-ensemble/merged_prediction_distance.geojson"
    # prediction_path = "./ensemble/predictions/prediction-ensemble/merged_prediction_similarity.geojson"
    # prediction_path = "./ensemble/predictions/prediction-ensemble/merged_prediction_average_new.geojson"
    # prediction_path = "./ensemble/predictions/prediction-model-06/merged_prediction.json"
    # prediction_path = "./ensemble/predictions/prediction-base-model/merged_prediction.json"

    # reference_path = "./evaluation/reference/building_building_.geojson"

    prediction_path = FLAGS.prediction_path
    reference_path = FLAGS.reference_path

     
    print("start evaluating...")
    start_time = time.time()

    # metrics path
    output_file = "evaluation_matrix.csv"
    header = ["Model", "IOU", "Filter", "TP", "FP", "FN", "predictions", "precision", "accuracy", "recall", "f1"]

    # with open(output_file,"w") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)


    # threshold
    COVERAGE_THRESHOLD = 0.8 # if the ratio of intersection area to prediction area higher than this threshold, it means the prediction is mostly covered by reference
    # IOU_range = np.arange(0.5, 0.8, 0.1)
    IOU_range = [0.5]
    # Filter_range = np.arange(0.04, 0.15, 0.01)
    Filter_range = [0.1]
    # print(IOU_range)
    # print(Filter_range)
    for IOU_THRESHOLD in IOU_range:
        for FILTER_THRESHOLD in Filter_range:
            print("Current evaluation threshold: ", IOU_THRESHOLD, FILTER_THRESHOLD)

            metrics = main_eval(prediction_path, reference_path, FILTER_THRESHOLD, IOU_THRESHOLD, COVERAGE_THRESHOLD)
            # metrics = metrics_statistic(results)

            model = "base"

            print("metrics: ", metrics)
            data = [str(model), IOU_THRESHOLD, FILTER_THRESHOLD, metrics["TP"], metrics["FP"], metrics["FN"], metrics["total_pred"], metrics["precision"], metrics["accuracy"], metrics["recall"], metrics["f1"]]
                            
            # with open(output_file, "a") as file:
            #     writer = csv.writer(file)
            #     writer.writerow(data)
        # file.write("{}  {:.2f}  {:.2f}  {}  {}  {}  {}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  \n".format(
        #     str(model), IOU_THRESHOLD, FILTER_THRESHOLD, metrics["TP"], metrics["FP"], metrics["FN"], metrics["total_pred"], metrics["precision"], metrics["accuracy"], metrics["recall"], metrics["f1"]
        #     ))
                

            print("time used: {}".format(time.time() - start_time))