gcloud ml-engine jobs submit training object_detection_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.9 \
    --job-dir=gs://ships-detections/models/mask_rcnn_inception_resnet_v2_atrous_coco_ship \
    --packages packages/object_detection-0.1.tar.gz,packages/slim-0.1.tar.gz,packages/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-central1 \
    --config cloud.yml \
    -- \
    --model_dir=gs://ships-detections/models/mask_rcnn_inception_resnet_v2_atrous_coco_ship \
    --pipeline_config_path=gs://ships-detections/models/mask_rcnn_inception_resnet_v2_atrous_coco_ship/googcl_pipeline.config