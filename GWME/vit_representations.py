# =======================================================================================================================================================
# vit_representations.py
# Author: Jiapan Wang
# Created Date: 01/06/2023
# Description: Inference of pre-trained DINO/ViT, and calculate attention weights.
# =======================================================================================================================================================

import zipfile
from io import BytesIO

import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Constants

RESOLUTION = 224
PATCH_SIZE = 16
GITHUB_RELEASE = "https://github.com/sayakpaul/probing-vits/releases/download/v1.0.0/probing_vits.zip"
FNAME = "probing_vits.zip"
MODELS_ZIP = {
    "vit_dino_base16": "Probing_ViTs/vit_dino_base16.zip",
    "vit_b16_patch16_224": "Probing_ViTs/vit_b16_patch16_224.zip",
    "vit_b16_patch16_224-i1k_pretrained": "Probing_ViTs/vit_b16_patch16_224-i1k_pretrained.zip",
}

# Data utilities

crop_layer = keras.layers.CenterCrop(RESOLUTION, RESOLUTION)
norm_layer = keras.layers.Normalization(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
)
rescale_layer = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)


def preprocess_image(image, model_type, size=RESOLUTION):
    # Turn the image into a numpy array and add batch dim.
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    # If model type is vit rescale the image to [-1, 1].
    if model_type == "original_vit":
        image = rescale_layer(image)

    # Resize the image using bicubic interpolation.
    resize_size = int((224 / 224) * size)
    image = tf.image.resize(image, (resize_size, resize_size), method="bicubic")

    # Crop the image.
    image = crop_layer(image)

    # If model type is DeiT or DINO normalize the image.
    if model_type != "original_vit":
        image = norm_layer(image)

    return image.numpy()


def load_image_from_url(url, model_type):
    # Credit: Willi Gierke
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    preprocessed_image = preprocess_image(image, model_type)
    return image, preprocessed_image

def load_image_from_local(path, model_type):
    # Credit: Willi Gierke
#     response = requests.get(url)
    image = Image.open(path)
    preprocessed_image = preprocess_image(image, model_type)
    return image, preprocessed_image

# # Load a model
# download models
# zip_path = tf.keras.utils.get_file(
#     fname=FNAME,
#     origin=GITHUB_RELEASE,
# )

# with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     zip_ref.extractall("./")

# os.rename("Probing ViTs", "Probing_ViTs")


def load_model(model_path: str) -> tf.keras.Model:
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        zip_ref.extractall("Probing_ViTs/")
    model_name = model_path.split(".")[0]

    inputs = keras.Input((RESOLUTION, RESOLUTION, 3))
    model = keras.models.load_model(model_name, compile=False)
    outputs, attention_weights = model(inputs, training=False)

    return keras.Model(inputs, outputs=[outputs, attention_weights])


# Mean attention distance
def compute_distance_matrix(patch_size, num_patches, length):
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights, model_type):
    num_cls_tokens = 2 if "distilled" in model_type else 1

    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length**2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token.
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # Sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # Now average across all the tokens

    return mean_distances


# Attention heatmaps
def attention_heatmap(attention_score_dict, image, num_heads, model_type="dino"):
    num_tokens = 2 if "distilled" in model_type else 1

    # Sort the Transformer blocks in order of their depth.
    attention_score_list = list(attention_score_dict.keys())
    attention_score_list.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)

    # Process the attention maps for overlay.
    w_featmap = image.shape[2] // PATCH_SIZE
    h_featmap = image.shape[1] // PATCH_SIZE
    attention_scores = attention_score_dict[attention_score_list[0]]

    # Taking the representations from CLS token.
    attentions = attention_scores[0, :, 0, num_tokens:].reshape(num_heads, -1)

    # Reshape the attention scores to resemble mini patches.
    attentions = attentions.reshape(num_heads, w_featmap, h_featmap)
    attentions = attentions.transpose((1, 2, 0))

    # Resize the attention patches to 224x224 (224: 14x16).
    attentions = tf.image.resize(
        attentions, size=(h_featmap * PATCH_SIZE, w_featmap * PATCH_SIZE)
    )
    return attentions

def generate_attention_map(img_path, vit_model, model_type):

    # Preprocess the same image but with normlization.
    # img_url = "https://dl.fbaipublicfiles.com/dino/img.png"
    # image, preprocessed_image = load_image_from_url(img_url, model_type="dino")

    image, preprocessed_image = load_image_from_local(img_path, model_type=model_type)

    # Grab the predictions.
    predictions, attention_score_dict = vit_model.predict(preprocessed_image)

    # Build the mean distances for every Transformer block.
    mean_distances = {
        f"{name}_mean_dist": compute_mean_attention_dist(
            patch_size=PATCH_SIZE,
            attention_weights=attention_weight,
            model_type=model_type,
        )
        for name, attention_weight in attention_score_dict.items()
    }

    # Get the number of heads from the mean distance output.
    num_heads = tf.shape(mean_distances["transformer_block_0_att_mean_dist"])[-1].numpy()
    # num_heads = 12

    # Print the shapes
    print(f"Num Heads: {num_heads}.")

    # De-normalize the image for visual clarity.
    in1k_mean = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])
    in1k_std = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])
    preprocessed_img_orig = (preprocessed_image * in1k_std) + in1k_mean
    preprocessed_img_orig = preprocessed_img_orig / 255.0
    preprocessed_img_orig = tf.clip_by_value(preprocessed_img_orig, 0.0, 1.0).numpy()

    # Generate the attention heatmaps.
    attentions = attention_heatmap(attention_score_dict, preprocessed_img_orig, num_heads, model_type)

    # Plot the maps.
    # fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13, 13))
    # img_count = 0

    # for i in range(3):
    #     for j in range(4):
    #         if img_count < len(attentions):
    #             axes[i, j].imshow(preprocessed_img_orig[0])
    #             print(attentions[..., img_count])
    #             axes[i, j].imshow(attentions[..., img_count], cmap="inferno", alpha=0.6)
    #             axes[i, j].title.set_text(f"Attention head: {img_count}")
    #             axes[i, j].axis("off")
    #             img_count += 1

    # print(np.shape(preprocessed_img_orig))
    # plt.imshow(preprocessed_img_orig[0])
    # plt.figure()

    return preprocessed_img_orig, attentions


def get_image_attention_weights(image_id, img_path, vit_model, output_dir="./", model_type="dino"):
    
    preprocessed_img_orig, attentions = generate_attention_map(img_path, vit_model, model_type)

    # average attention of 12 heads attention
    sum_attentions = tf.constant(np.zeros([RESOLUTION, RESOLUTION], dtype='float32'))
    for i in range(12):
        sum_attentions = tf.add(sum_attentions, attentions[..., i])
    avg_attention = sum_attentions/12
    
    plt.imshow(preprocessed_img_orig[0])
    plt.imshow(avg_attention, cmap="inferno", alpha=0.5)
    plt.title(f'{image_id}')
    plt.savefig(f'{output_dir}/{image_id}_attention_map.png')

    attention_map = avg_attention.numpy()
    # print(attention_map)
    attention_patches = []
    attention_weights = []

    # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(13, 13))
    # img_count = 0

    for i in range(3):
        for j in range(3):
            attention_patch = attention_map[i*74+1:(i+1)*74+1, j*74+1:(j+1)*74+1]
            attention_patches.append(attention_patch)
            attention_weights.append(np.sum(attention_patch))
            # print("patch",i,j,np.shape(attention_patch))

    #         plt.figure()
            # axes[i, j].imshow(attention_patch, interpolation='none')
            # axes[i, j].title.set_text(f"patch: {i},{j}")
            # axes[i, j].axis("off")
            
            
    # print(np.shape(attention_patches))
    print(attention_weights)

    center_weight = attention_weights[4]
    for i in range(len(attention_weights)):
        attention_weights[i] =  attention_weights[i]/center_weight
        
    # attention_weights.pop(4)
    print(attention_weights)

    # im = Image.fromarray(attention_map)
    # im.save("attention_map.png")
    print("attention_map...done!")

    return attention_weights

if __name__ == '__main__':

    img_path = "merged_02.png"
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    model_type = "dino"

    # Load the model.
    if model_type == "dino":    
        vit_dino_base16 = load_model(MODELS_ZIP["vit_dino_base16"])
    else:
        print("Didn't find the model.")
    print("Model loaded.")

    attention_weights = get_image_attention_weights(image_id, img_path, vit_dino_base16, model_type)
    


    # resized_img = tf.image.resize(preprocessed_img_orig[0], (768,768), method="bicubic")
    # plt.imshow(resized_img)
    # plt.figure()
    # plt.show()

