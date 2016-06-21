#!/bin/bash

wget "http://www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/sbir_cvpr2016.zip"
unzip -o sbir_cvpr2016.zip
mkdir data
mv sbir_cvpr2016_release/* data/
rm -r sbir_cvpr2016_release

