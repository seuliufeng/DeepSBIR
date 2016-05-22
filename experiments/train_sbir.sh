#!/bin/bash


python tools/sbir_train_net.py --gpu 0 \
--solver ./experiments/solver.prototxt \
--output ./snapshot/ \      # make sure you create a folder named 'snapshot' under your $SBIR_ROOT folder
--iters 25000 \
--snapstep 500 \
--weights ./experiments/init/sketchnet_init.caffemodel
