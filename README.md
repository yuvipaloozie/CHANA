# CHANA (Cell Histology Automated Nuclei Analyzer)

CHANA is an end-to-end image analysis tool that utilizes a convolutional neural network to automatically classify multinucleated osteoclast cells based on high-resolution micrscope images. 
The model utilizes a heavily regularized U-Net architecture, which allows for pixel-level semantic segmentation of the images into the necessary classes (i.e. identifiying osteoclasts in images containing other , undifferentiated cells).
This repository consist of multiple different programs, some of which were for the training of the model, and the others which belong to the actual end tool for image counting:
* CHANA Preprocessing - Image preprocessing script which prepares high-res .tif images into a "digestible" format for the model
* CHANA CNN
* CHANA Pipeline


