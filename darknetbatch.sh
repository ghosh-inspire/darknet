#!/bin/bash

#SBATCH -p lyceum

## mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

## send mail to this address
#SBATCH --mail-user=sg1g18@soton.ac.uk

#SBATCH --time=60:00:00          # walltime
#SBATCH --dependency=afterok:55180

##conda activate darknet
##conda install -c anaconda pango
##conda install -c menpo opencv3
##export PKG_CONFIG_PATH=/lyceum/sg1g18/conda/envs/darknet/lib/pkgconfig/
export LD_LIBRARY_PATH=/lyceum/sg1g18/conda/envs/darknet/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lyceum/sg1g18/comp6200/common/libpng/

## VOC
##./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg backup/yolov3-voc.backup -gpus 0
##./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg /scratch/sg1g18/weights/darknet53.conv.74 -gpus 0

## COCO
##./darknet detector train cfg/coco.data cfg/yolov3-tiny.cfg /scratch/sg1g18/weights/darknet53.conv.74 -gpus 0
##./darknet detector train cfg/coco.data cfg/tinyv3.cfg ../../data/weights/yolov3-tiny.weights
##./darknet detector train cfg/coco.data cfg/yolov3-tiny.cfg -gpus 0
#./darknet classifier train cfg/cifar.data cfg/tinyv3.cfg
#./darknet classifier train cfg/cifar.data cfg/tinyv3.cfg backup/tinyv3.backup
#./darknet classifier train cfg/cifar.data cfg/tiny.cfg
#./darknet detector train cfg/voc.data cfg/yolov3-tiny-exit.cfg #slurm-48836.out
#./darknet detector train cfg/voc.data cfg/yolov3-tiny-exit_full.cfg #slurm-48993.out, multi exit, voc
#./darknet detector train cfg/voc.data cfg/yolov3-tiny.cfg #slurm-49001.out, yolov3-tiny orig, voc
#./darknet detector train cfg/voc.data cfg/yolov3-tiny-exit.cfg #slurm-49832.out, random classifier label, voc
#./darknet detector train cfg/coco.data cfg/yolov3-tiny-yexit-coco.cfg #slurm-52223.out, yexit, a6-012, coco
#./darknet detector train cfg/coco.data cfg/yolov3-tiny-yexit-sexit-rcls-a6-012-coco.cfg #slurm-55144.out, yexit, sexit, a6-012, coco
#./darknet detector train cfg/coco.data cfg/yolov3-tiny-yexit-sexit-rcls-a9-coco.cfg #slrum-55176.out, yexit, sexit, a9, coco
#./darknet detector train cfg/coco.data cfg/yolov3-tiny-yexit-sexit-rcls-a6-345-coco.cfg #slrum-55181.out, yexit, sexit, a6-345, coco

