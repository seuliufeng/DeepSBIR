import os
import numpy as np
from time import time
from scipy.io import loadmat, savemat
from sbir_config import load_model_config
import scipy.spatial.distance as ssd
from skimage.transform import resize
from sbir_util.batch_manager import MemoryBlockManager


NUM_VIEWS = 10


def imresize(im, input_dim, force_color=False):
    def is_single_channel(im):
        return im.ndim == 2 or (im.ndim == 3 and im.shape[-1] == 1)
    if im.shape[0] != input_dim:
        im = resize(im, (input_dim, input_dim), preserve_range=True)
    if force_color and is_single_channel(im):
        im = np.tile(im.reshape([input_dim, input_dim, 1]), [1, 1, 3])
    return im


def do_multiview_crop(data, cropsize):
    if len(data.shape) == 2: # single sketch
        data = data[np.newaxis, np.newaxis, :, :]  # nxcxhxw
    elif len(data.shape) == 3: # sketch
        n, h, w = data.shape
        data = data.reshape((n, 1, h, w))
    n, c, h, w = data.shape
    xs = [0, 0, w-cropsize, w-cropsize]
    ys = [0, h-cropsize, 0, h-cropsize]
    batch_data = np.zeros((n*10, c, cropsize, cropsize), np.single)
    y_cen = int((h - cropsize) * 0.5)
    x_cen = int((w - cropsize) * 0.5)
    for i in xrange(n):
        for (k, (x, y)) in enumerate(zip(xs, ys)):
            batch_data[i*10+k, :, :, :] = data[i, :, y:y+cropsize, x:x+cropsize]
        # center crop
        batch_data[i*10+4, :, :, :] = data[i, :, y_cen:y_cen+cropsize, x_cen:x_cen+cropsize]
        for k in xrange(5):  # flip
            batch_data[i*10+k+5, :, :, :] = batch_data[i*10+k, :, :, ::-1]
    return batch_data


def get_feature_dims(net_, layer_name):
    wblob = net_.blobs[layer_name]
    return wblob.channels*wblob.height*wblob.width


def reshape_feature(target):
    if len(target.shape) == 4:
        n, c, h, w = target.shape
        return target.reshape(n, c * h * w)
    elif len(target.shape) == 2:
        return target
    else:
        raise Exception('unknown dim')


def reshape_multiview_features(feats):
    n, c = feats.shape
    feats = feats.reshape(n / NUM_VIEWS, NUM_VIEWS, c)
    return feats


def extract_features(net_, config):
    db = MemoryBlockManager(config.DB_PATH, False)
    if config.verbose:
        print 'extracting features of %s' % config.dataset
    t = time()
    num_samples = 10 * db.num_samples
    feats = np.zeros((num_samples, get_feature_dims(net_, config.feat_layer)), np.single)
    idx = 0
    while not db.eof():
        batch_data, _ = db.pop_batch(config.batchsize) # batchsize 128
        transformed = do_multiview_crop(batch_data, config.crop_dim)
        transformed = transformed - config.mean_val
        num, chns, rows, cols = transformed.shape
        net_.blobs['data'].reshape(*(num, chns, rows, cols))
        output = net_.forward(data=transformed.astype(np.float32, copy=False))
        target = output[config.feat_layer]
        feats[idx:idx+num, ::] = reshape_feature(target)
        idx += num
    if config.verbose:
        print 'feature computation completed (%0.2f sec.)' % (time()-t)
    return feats


def compute_view_specific_distance(sketch_feats, image_feats):
    sketch_feats = reshape_multiview_features(sketch_feats)
    image_feats = reshape_multiview_features(image_feats)
    num_sketches, num_images = sketch_feats.shape[0], image_feats.shape[0]
    multi_view_dists = np.zeros((NUM_VIEWS*2, num_sketches, num_images))
    for i in xrange(NUM_VIEWS):
        multi_view_dists[i, ::] = ssd.cdist(sketch_feats[:, i, :], image_feats[:, i, :])
        multi_view_dists[i+NUM_VIEWS, ::] = ssd.cdist(sketch_feats[:, i, :], image_feats[:, -i, :])
    return multi_view_dists


def cache_features(net_, dataset):
    config = load_model_config(dataset)
    feat_path = config.FEAT_PATH
    if os.path.exists(feat_path):
        return loadmat(feat_path)['feats']
    # make dir
    feat_rt = os.path.split(feat_path)[0]
    if not os.path.exists(feat_rt):
        os.makedirs(feat_rt)
    # extract feature
    print 'caching features'
    feats = extract_features(net_, config)
    savemat(feat_path, {'feats': feats})
    return feats


def sketch_retrieval(net_, im, image_feats):
    config = load_model_config()
    im = imresize(im, config.input_dim)
    im = im.astype(np.single) - config.mean_val
    transformed = do_multiview_crop(im, config.crop_dim)
    num, chns, rows, cols = transformed.shape
    # reshape data layer
    net_.blobs['data'].reshape(*(num, chns, rows, cols))
    # forward
    output = net_.forward(data=transformed.astype(np.float32, copy=False))
    target = reshape_feature(output[config.feat_layer].copy())
    # get retrieval results
    multiview_dists = compute_view_specific_distance(target, image_feats)
    ave_dist = multiview_dists.mean(axis=0)
    return ave_dist.flatten().argsort()  # return ranklist
