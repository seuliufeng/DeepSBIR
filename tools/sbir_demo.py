import caffe
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from sbir_util.smts_api import SMTSApi
from sbir_config import load_model_config
from sbir_retrieval import sketch_retrieval, cache_features, imresize


class SBIRWarpper:
    def __init__(self, model_path, dataset):
        self.net_ = None
        self.dataset_ = None
        self.image_feats_ = None
        self.config_ = load_model_config(dataset)
        self.load_dataset(dataset)
        self.set_cnn_model(model_path, self.config_.deploy_file)
        self.load_cached_features()

    def load_dataset(self, dataset_name):
        self.dataset_ = SMTSApi(dataset_root=self.config_.DATASET_ROOT,
                                name=dataset_name)

    def load_cached_features(self):
        self.image_feats_ = cache_features(self.net_, self.dataset_.name)

    def set_cnn_model(self, weight_path, deploy_file):
        caffe.set_device(self.config_.gpu_id)
        caffe.set_mode_gpu()
        self.net_ = caffe.Net(deploy_file, weight_path, caffe.TEST)

    def run_retrieval(self, im):
        ranklist = sketch_retrieval(self.net_, im, self.image_feats_)
        return self.dataset_.get_image_pathes(ranklist, 'test')


def vis_retrieval(query, results, true_match, n_imgs_per_row=5):
    DIM = 256
    nrows = (len(results) + n_imgs_per_row - 1) / n_imgs_per_row  # ceil
    canvas = 255 * np.ones((DIM*nrows, DIM*(n_imgs_per_row+1), 3), np.uint8)  # white canvas

    def paint(row, col, patch):
        x1, y1, x2, y2 = col*DIM, row*DIM, (col+1)*DIM, (row+1)*DIM
        canvas[y1:y2, x1:x2, :] = patch
    query = imresize(query, DIM, force_color=True)
    paint(0, 0, query)
    for i, impath in enumerate(results):
        im = imread(impath)
        if impath.endswith(true_match):  # true match, draw green boarder
            color = np.array([0, 255, 0]).reshape([1, 1, 3])
            im[:5,:,:] = im[-5:,:,:] = color
            im[:,:5,:] = im[:,-5:,:] = color
        paint(i/n_imgs_per_row, i%n_imgs_per_row+1, im)
    imshow(canvas)
    plt.axis('off')
    plt.show()


def sbir_demo():
    def test(imname, true_match, model_path, dbname, topK=10):
        model = SBIRWarpper(model_path, dbname)
        im = imread(imname)
        results = model.run_retrieval(im)
        print 'retrieval results for sketch: %s' % imname
        for i in xrange(topK):
            print '%s' % results[i]
        vis_retrieval(im, results[:topK], true_match)
    # test shoes
    test('test_shoes_370.png', '370.png', 'data/models/shoes.caffemodel', 'shoes')
    # test chairs
    test('test_chairs_294.png', '294.png' ,'data/models/chairs.caffemodel', 'chairs')


if __name__ == "__main__":
    sbir_demo()
