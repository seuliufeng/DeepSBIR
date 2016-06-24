# Sketch Me *That* Shoe


###Introduction

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

	
####Contents

1. [License](#license)

2. [Installation](#installation)

3. [Run the demo](#run-the-demo)

4. [Re-training the model](#re-training-the-model)

5. [Extra comment](#extra-comment)

###License
**MIT License**

###Installation
1. Download the repository

	```shell
	git clone git@github.com:seuliufeng/DeepSBIR.git
	```

2. Build Caffe and pycaffe

	a. Go to root folder of this project

	b. modify the path in Makefile.config, to use this code, you have to compile with python layer
	```make
	  WITH_PYTHON_LAYER := 1
	```

	c. Compile caffe 
	```shell make –j32 && make pycaffe```

3. Configure environment variable. Modify the path in bashsbir to your own path, and run
	```shell
	source bashsbir
	```

###Run the demo

1. To run the demo, please first download our database and models. Go to the root folder of this project, and run

	``` shell
	chmod +x download_data.sh
	./download_data.sh
	```

**Note:** You can also download them manually from our project page: http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html


2. Run the demo:

```shell
python $SBIR_ROOT/tools/sbir_demo.py
```

###Re-training the model
1. Go to the root folder of this project

	``` shell
	cd $SBIR_ROOT
	```

2. Run the command

```shell
./experiments/train_sbir.sh
```
**Note:** Please make sure the initial model ‘/init/sketchnet_init.caffemodel’ be under the folder experiments/. This initial model can be downloaded from our project page. 
	
###Extra comment
1. All provided models and codes are optimised version. And our latest result is shown below:

   | Dataset |	acc.@1	|  acc.@10 |   %corr.  |
   |:-------:|:--------:| --------:| ---------:|
   | Shoes   | 53.91%	| 91.30%   | 72.29%    |
   | Chairs  | 72.16%	| 98.97%   | 74.36%    |

2. This project used codes of the following project:

   [Caffe trainnet python wrapper and python data layer](https://github.com/rbgirshick/fast-rcnn)

   [L2 normalization layer](https://github.com/happynear/caffe-windows)
   
   [Triplet loss](http://blog.csdn.net/tangwei2014/article/details/46812153)

