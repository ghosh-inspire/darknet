#!/bin/bash

#SBATCH -p lyceum

## mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

## send mail to this address
#SBATCH --mail-user=sg1g18@soton.ac.uk

#SBATCH --time=60:00:00          # walltime
#SBATCH --dependency=afterok:24830

##conda activate darknet
##conda install -c anaconda pango
##conda install -c menpo opencv3
##export PKG_CONFIG_PATH=/lyceum/sg1g18/conda/envs/darknet/lib/pkgconfig/
export LD_LIBRARY_PATH=/lyceum/sg1g18/conda/envs/darknet/lib/

## VOC
##./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup -gpus 0
##./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg /scratch/sg1g18/weights/darknet53.conv.74 -gpus 0

## COCO
##./darknet detector train cfg/coco.data cfg/yolov3-tiny.cfg /scratch/sg1g18/weights/darknet53.conv.74 -gpus 0
##./darknet detector train cfg/coco.data cfg/tinyv3.cfg ../../data/weights/yolov3-tiny.weights
##./darknet detector train cfg/coco.data cfg/yolov3-tiny.cfg -gpus 0
./darknet classifier train cfg/cifar.data cfg/tinyv3.cfg
