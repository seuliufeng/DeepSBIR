#!/bin/bash

# download data
wget "http://www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/sbir_cvpr2016.tar"
tar -xvf sbir_cvpr2016.tar
rsync -a sbir_cvpr2016_release/ data/
rm -r sbir_cvpr2016_release
rm sbir_cvpr2016.tar

# remove features in data/feats to avoid feature conflicts caused by different models
rm data/feats/*

# download model
wget "http://www.eecs.qmul.ac.uk/%7Eqian/Qian%27s%20Materials/sbir_models.tar"
tar -xvf sbir_models.tar
rsync -a models/ data/models
rm -r models
rm sbir_models.tar
