# GWME

Geographical Weighted Model Ensemble (GWME)

## Utils

### train, inference, export and evaluation

- Train: *./GWME/model_main_tf2.py*
- Inference: *./GWME/inference_main.py*, *./GWME/prediction_to_geojson.py*
- Export: *./exporter_main_v2.py*
- Evaluation: *./GWME/evaluation_geojson.py*

### calculating weights

- *./GWME/calculate_weights.py*
- *./GWME/vit_representations.py*
- *./GWME/merge_image_patch.py*

### Ensemble

- *./GWME/ensemble.py*

## Case Study

- predictions
- Probing_ViTs: pre-trained ViT
- sample_images: satellite image tiles from reference area and target area
- ViT_sample_images: image candidates for calculating attention weights
- all_weights.json: tile-wise weights dictionary