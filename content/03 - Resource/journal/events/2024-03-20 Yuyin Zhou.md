---
title: Yuyin Zhou
allDay: false
startTime: 12:00
endTime: 13:00
date: 2024-03-20
completed: null
---
# Medical Imaging x AI: In pursuit of foundation model in MI

## Motivation
Quicker, serve as a second reader

## Limitation in current SOTA
- Hard to perform on multimodal data
- Cannot perform well on data that has not been seen

## Areas
### From customized model to transformer
#### The Felix Project
Use U-Net to segment pancreatic tumor: use a coarse u-net to segment the ROI, and use a fine-grained unet to segment tumor on the ROI
#### TransUnet 
Insert transformer encoder layers to the bottleneck of the original U-Net
#### Segment anything in medical images
### Building Multimodal models 
#### Contrastive Learnign og medical visual representations from paired images and text
- Align Vision with language
#### Multi-granularity cross-modal alignment for generalized medical l visual representation 
- alignement in 3 granularities 
#### BiomedGPT
Self-supervised pretraining 

#### Medical Vision Generalist
### deployment in real-word system
