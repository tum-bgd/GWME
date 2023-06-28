# =======================================================================================================================================================
# calculate_weights.py
# Author: Jiapan Wang
# Created Date: 01/06/2023
# Description: Calculate the weight of distance, image correlation and attention between the reference area and the target area.
# =======================================================================================================================================================

from math import sin, cos, acos, atan2, radians, pi, atan, exp
from sklearn.preprocessing import normalize
import numpy as np
from PIL import Image
import time
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import random
import json
from merge_image_patch import merge_images
from vit_representations import get_image_attention_weights, load_model, MODELS_ZIP

IMAGE_DIR = './sample_images/'

def get_center_latlon(lon1, lat1, lon2, lat2):

    center_lat = abs(lat1 - lat2)/2 + min(lat1, lat2)
    center_lon = abs(lon1 - lon2)/2 + min(lon1, lon2)

    return center_lon, center_lat


def calculate_distance(lon1, lat1, lon2, lat2):
    """
    Calculate distance in meters between two latitude, longitude points.

    Law of cosines: d = acos( sin φ1 ⋅ sin φ2 + cos φ1 ⋅ cos φ2 ⋅ cos Δλ ) ⋅ R
    ACOS( SIN(lat1)*SIN(lat2) + COS(lat1)*COS(lat2)*COS(lon2-lon1) ) * 6371000
    """
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    distance = acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos((lon2 - lon1))) * R * 1000
    # print("distance", distance) # m

    return distance

def distance_weights(c_lon_target, c_lat_target):
    """
    center_box: bbox [left, top, right, bottom] of the target area.
    """

    c_lat = np.zeros(8)
    c_lon = np.zeros(8)
    distance = np.zeros(8)

    bbox_list = [
        [10.1808867479861433,5.6779222981920654,10.1854357002290943,5.6815842067768036],
        [10.2044794916899786,5.6874603256778036,10.2081243554005656,5.6910385317555923],
        [10.2369822404324680,5.6739091434510103,10.2427961658346156,5.6787338574077060],
        [10.1758264495190556,5.6589049871613826,10.1808537742712719,5.6624908025580210],
        [10.2405342704539137,5.6561765898082879,10.2458976227248382,5.6612797033758442],
        [10.1783557870918955,5.6422238384631669,10.1850979479314638,5.6475389476528628],
        [10.2071616243839944,5.6417252758946166,10.2115071541707554,5.6463194782201533],
        [10.2613935439247506,5.6314816109111181,10.2656648067598848,5.6353477536127228]
    ] 

    

    for i in range(0, 8):
        # print("bbox", bbox_list[i][0], bbox_list[i][1], bbox_list[i][2], bbox_list[i][3])
        c_lon[i], c_lat[i] = get_center_latlon(bbox_list[i][0], bbox_list[i][1], bbox_list[i][2], bbox_list[i][3])
        distance[i] = calculate_distance(c_lon[i], c_lat[i], c_lon_target, c_lat_target)
    

    # print("center:", c_lat, "\n", c_lon)
    # print("distance:", distance)
    # print("inverse distance", 1.0/distance)

    # distance weights
    # print("Distance weights:")
    # weights = normalize_weights(distance)

    # inverse distance weights
    print("Inverse distance weights:")
    inv_weights = normalize_weights(1.0/distance)

    return distance, inv_weights

# Load images
def load_images(path):
    
    # Get the list of all files and directories
    dir_list = os.listdir(path)
    # print("Files and directories in '", path, "' :")
    # prints all files
    # print(dir_list)
    
    filenames = dir_list
    image_paths = []
    for filename in filenames:
#         image_path = tf.keras.utils.get_file(fname=filename,
#                                             origin=path + filename,
#                                             untar=False)
        image_path = pathlib.Path(path+filename)
        image_paths.append(str(image_path))

    return image_paths

def calculate_image_similarity(img1, img2, channel):
    
    similarity = 0

    for i in range(channel):
        
        hist_similarity = cv2.compareHist(cv2.calcHist([img1], [i], None, [256], [0, 256]), cv2.calcHist([img2], [i], None, [256], [0, 256]), cv2.HISTCMP_CORREL)
        similarity += hist_similarity
        # print(i, "similarity", hist_similarity)
    #     hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    #     hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    # similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # gray_similarity = cv2.compareHist(cv2.calcHist([img1_gray], [0], None, [256], [0, 256]), cv2.calcHist([img2_gray], [0], None, [256], [0, 256]), cv2.HISTCMP_CORREL)
    similarity = similarity / channel
    # print("similarity", similarity)
    # print("gray_similarity", gray_similarity)
    return similarity

# def image_similarity_weights():

#     ref_dirs = os.listdir(IMAGE_DIR)
    
#     # ref_paths = load_images(IMAGE_DIR)
#     target_dir = ref_dirs.pop()
#     print(target_dir)

#     # target image path list
#     target_image_path = IMAGE_DIR+target_dir+"/"
#     target_image_paths = load_images(target_image_path)
#     # print(target_image_paths)

#     average_similarity = []

#     for i, ref_dir in enumerate(ref_dirs):
#         print("Image similarity between {} and {}".format(ref_dir, target_image_path))
#         # reference image path list
#         ref_image_paths = load_images(IMAGE_DIR+ref_dir+"/")
#         # print("ref",ref_image_paths)
#         length = len(ref_image_paths)
#         # print(length)

#         # pick random image samples from target area 
#         target_image_samples = random.sample(target_image_paths, length)
#         # print("samples", target_image_samples)

#         similarity = 0

#         for j, image_path in enumerate(ref_image_paths):

#             # print(i, image_path)
#             # print("target", target_image_samples[i])

#             img1 = cv2.imread(image_path)
#             img2 = cv2.imread(target_image_samples[j])

#             similarity += calculate_image_similarity(img1, img2, 3)

#         average_similarity.append(similarity/len(ref_image_paths))

#         # print("Average Similarity between {} and {} is {} \n".format(ref_dir, target_image_path, average_similarity[i]))     

#         # img1 = cv2.imread('img5.png')
#         # img2 = cv2.imread('img2.png')

#     print("Average Similarity List", average_similarity)
#     similarity_weights = normalize_weights(np.array(average_similarity))

#     return average_similarity, similarity_weights



def normalize_weights(weight):

    norm_weights = normalize(weight[:,np.newaxis], axis=0, norm='l1').ravel()
    print("weights:", norm_weights, "\nsum of weights:",sum(norm_weights), "\n")

    return norm_weights

def pixel_coords_zoom_to_lat_lon(PixelX, PixelY, zoom):
    MapSize = 256 * pow(2, zoom)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon = 360 * x
    lat = 90 - 360 * atan(exp(-y * 2 * pi)) / pi

    return lon, lat

def parse_tile_name(name):
    zoom, TileX, TileY = [int(x) for x in name.split(".")]
    return TileX, TileY, zoom

def tile_to_ref_average_similarity_weights(tile_id):

    ref_dirs = os.listdir(IMAGE_DIR)
    
    target_dir = ref_dirs.pop()
    # print(ref_dirs)

    target_image_path = IMAGE_DIR + target_dir + "/" + tile_id + ".png"
    tile_image = cv2.imread(target_image_path)

    # print("target tile image: ", tile_image)
    average_similarity = []

    for ref_dir in ref_dirs:

        ref_image_path = IMAGE_DIR + ref_dir + "/"
        ref_image_paths = load_images(ref_image_path)
        # print(ref_image_paths)

        similarity = 0

        for image_path in ref_image_paths:
            
            ref_image = cv2.imread(image_path)
            similarity += calculate_image_similarity(tile_image, ref_image, 3)
            # print("sum similarity: ", similarity)

        average_similarity.append(similarity/len(ref_image_paths))

        # print("Average Similarity between {} and {} is {} \n".format(ref_dir, target_image_path, average_similarity)) 

    norm_average_similarity = normalize_weights(np.array(average_similarity))
    print("normalized averaged similarity weight is", norm_average_similarity)

    return norm_average_similarity

def tile_to_ref_distance_weights(tile_id):
    
    tileX, tileY, zoom = parse_tile_name(tile_id)

    # print("tile id: ", tile_id)
    # print(tileX, tileY, zoom)

    c_pixelX = tileX * 256 + 127
    c_pixelY = tileY * 256 + 127

    c_lon, c_lat = pixel_coords_zoom_to_lat_lon(c_pixelX, c_pixelY, zoom)

    # print(c_lon, c_lat)

    # distance weights
    distance, distance_weight = distance_weights(c_lon, c_lat)

    return distance_weight

def tile_to_ref_attention_weights(tile_id, vit_model):

    target_dir = os.listdir(IMAGE_DIR).pop()
    target_image_path = IMAGE_DIR + target_dir + "/" + tile_id + ".png"
    tile_image = cv2.imread(target_image_path)

    # replace center image patch
    tile_image_new_file = './ViT_sample_images/' + "5.png"
    cv2.imwrite(tile_image_new_file, tile_image)

    vit_sample_image_dir = './ViT_sample_images/'

    attention_maps_dir = './attention_maps/attention_map'
    if not os.path.exists(attention_maps_dir):
        os.makedirs(attention_maps_dir)

    merged_image_dir = './attention_maps/merged_image'
    if not os.path.exists(merged_image_dir):
        os.makedirs(merged_image_dir)

    # merge 9 image patches to one
    merged_image = merge_images(vit_sample_image_dir)
    im = Image.fromarray(merged_image, 'RGB')
    im.save(f"{merged_image_dir}/{tile_id}_merged.png")
    print(f"merging {tile_id}...done!")

    # generate attention map and weights
    img_path = merged_image_dir + "/" + tile_id + "_merged.png"
    # image_id = os.path.splitext(os.path.basename(img_path))[0]
    model_type = "dino"

    attention_weights = get_image_attention_weights(tile_id, img_path, vit_model, attention_maps_dir, model_type)
    # delete weight of center patch
    attention_weights.pop(4)
    norm_attention_weights = normalize_weights(np.array(attention_weights))

    return norm_attention_weights

def tile_to_ref_weights():

    target_tile_image_dir = IMAGE_DIR + "target/"
    tile_dir = os.listdir(target_tile_image_dir)
    # print("tile paths: ", tile_dir)

    weights_dict_list = []

    # Load the model. 
    vit_model = load_model(MODELS_ZIP["vit_dino_base16"])
    print("Model loaded.")

    for i, tile_name in enumerate(tile_dir):
        
        # image id
        tile_id = os.path.splitext(os.path.basename(tile_name))[0]

        print("\nCalculating weights for {}, {}\n".format(i, tile_id))

        # # distance weights
        distance_weights = tile_to_ref_distance_weights(tile_id)

        # # image similarity weights
        similarity_weights = tile_to_ref_average_similarity_weights(tile_id)

        # ViT-DINO attention weights
        attention_weights = tile_to_ref_attention_weights(tile_id, vit_model)

        # weight dictionary for each tile
        new_weight = {
            "tile_id": tile_id,
            "inverse_distance_weights": distance_weights.tolist(), 
            "image_similarity_weights": similarity_weights.tolist(),
            "attention_map_weights": attention_weights.tolist(),
        }
        weights_dict_list.append(new_weight)

        print("\n",tile_name, new_weight)
        
        # break
    
    # print("weight dict list: ", weights_dict_list)
    return weights_dict_list



if __name__ == '__main__':


    # print("distance weights:")
    # c_box = [10.1926316905051912,5.6555060217455528,10.2352523789601566,5.6722064157053831]
    # c_lon_target, c_lat_target = get_center_latlon(c_box)
    # distance, distance_weight = distance_weights(c_lon_target, c_lat_target)
    print("Start calculating weights ...")
    start_time = time.time()
    weight_dict_list = tile_to_ref_weights()
    # Writing to json file
    with open("all_weights.json","w", encoding='utf-8') as file:
        json.dump(weight_dict_list, file)
    
    print("done. time used: {}".format(time.time()-start_time))
    # # Writing to file
    # with open("weights.txt", "a") as file:
    #     # Writing data to a file
    #     file.write("image similarity: \n{}\n".format(similarity))
    #     file.write("image similarity weights: \n{}\n".format(similarity_weight.tolist()))
    #     file.write("distance: \n{}\n".format(distance.tolist()))
    #     file.write("distance weights: \n{}\n".format(distance_weight.tolist()))
    #     file.write("==================================================================================================================================\n")