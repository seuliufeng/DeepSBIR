Introduction

This repository contains the code for the CPVR paper ‘Sketch Me That Shoe’, which is a deep learning based implementation of fine-grained sketch-based image retrieval. 

For more details, please visit our project page:
http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html

And if you use the code for your research, please cite our paper:

	@inproceedings{qian2016,
	    Author = {Qian Yu, Feng Liu, Yi-Zhe Song, Tao Xiang, Timothy M. Hospedales and Chen Change Loy},
	    Title = {Sketch Me That Shoe},
	    Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	    Year = {2016}
	}

	
Contents

1. Requirements: software

2. License

3. Installation

4. Run the demo

5. Re-training the model

6. Extra comment

Installation

1. Download and unzip repository

2. Build Caffe and pycaffe

		a. Go to root folder of this project
		b. make –j32 && make pycaffe

3. Configure environment variable. Modify the path in bashsbir to your own path, and run

		source bashsbir
		
Run the demo

1. To run the demo, please first download our database and models from our project webpage:
	http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html

	Note: Make sure these four folders ‘sbir_cvpr2016, ‘models’, ‘feats’ and ‘dbs’ are under the data folder.

2. Run the demo:

		python $SBIR_ROOT/tools/sbir_demo.py
		
Re-training the model

1. cd $SBIR_ROOT

2. Run the command

		./experiments/train_sbir.sh	
		
	Note: Please make sure the initial model ‘/init/sketchnet_init.caffemodel’ be under the folder experiments/. This initial model can be downloaded from our project webpage. 
	
Extra comment

1. All provided models and codes are optimised version. And our latest result is shown below:
   
   Shoes dataset: 
		
	acc.@1: 53.91%	acc.@10: 91.3%	%corr.: 72.29%

   Chairs dataset: 
		
	acc.@1: 72.16%	acc.@10: 98.97%	%corr.: 74.36%

2. This project used codes of the following project:

Caffe trainnet python wrapper and python data layer:
https://github.com/rbgirshick/fast-rcnn

L2 normalization layer:
https://github.com/happynear/caffe-windows

Triplet loss:
http://blog.csdn.net/tangwei2014/article/details/46812153

