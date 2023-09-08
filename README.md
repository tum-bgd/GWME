# Rethink Geographical Generalizability with Unsupervised Self-Attention Model Ensemble: A Case Study of OpenStreetMap Missing Building Detection in Africa

## Abstract:

In this work, we proposed Geographical Weighted Model Ensemble (GWME), an unsupervised model ensemble method to improve the geographical generalizability of GeoAI models, with a case study of cross-country OpenStreetMap (OSM) missing building detection in sub-Saharan Africa. Moreover, we compare four unsupervised model ensemble weighting strategies: 1) Average weighting, 2) Image similarity weighting, 3) Geographical distance weighting, and 4) Self-attention-based weighting. One can find the source code as follows.



## Build:

```
docker build . -t gwme:<TAG>
docker run -it --gpus all --name gwme -p 8888:8888 -p 6006:6006 --mount type=bind,source="$(pwd)",target=/app gwme:<TAG>

jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --debug
```

## Utils:

### Train, inference, export and evaluation:

- Train: *./GWME/model_main_tf2.py*
- Inference: *./GWME/inference_main.py*, *./GWME/prediction_to_geojson.py*
- Export: *./exporter_main_v2.py*
- Evaluation: *./GWME/evaluation_geojson.py*

### Calculating weights:

- *./GWME/calculate_weights.py*
- *./GWME/vit_representations.py*
- *./GWME/merge_image_patch.py*

### Model ensemble:

- *./GWME/ensemble.py*

## Case Study

- predictions
- Probing_ViTs: pre-trained ViT
- sample_images: satellite image tiles from reference area and target area
- ViT_sample_images: image candidates for calculating attention weights
- all_weights.json: tile-wise weights dictionary

## Citation

Hao Li. Jiapan Wang, Johann Maximilian Zollner, Gengchen Mai, Ni Lao, and Martin Werner. 2023. Rethink Geographical Generalizability with Unsupervised Self-Attention Model Ensemble: A Case Study of OpenStreetMap Missing Building Detection in Africa.

## Contact

Hao Li: [hao_bgd.li@tum.de](mailto:hao_bgd.li@tum.de)  
Hao Li is with the Technische Universität München, Professur für Big Geospatial Data Management, Lise-Meitner-Str. 9, 85521 Ottobrunn


