#!/bin/bash

# download data
wget "http://www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/sbir_cvpr2016.tar"
tar -xvf sbir_cvpr2016.tar
rsync -a sbir_cvpr2016_release/ data/
rm -r sbir_cvpr2016_release
rm sbir_cvpr2016.zip

# download model
wget "http://www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/sbir_models.zip"
unzip -o sbir_models.zip
rsync -a models/ data/models
rm -r models
rm sbir_models.zip
