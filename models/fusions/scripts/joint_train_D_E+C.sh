python run_fusions.py \
--gpu_device 0 \
--modalities EHR+CXR \
--mode train \
--stage finetune_joint \
--data_root ../D_E+C+G \
--save_dir ./results/fusions/joint/D_E+C \
--layers 1 \
--vision_backbone resnet50 \
--pretrained