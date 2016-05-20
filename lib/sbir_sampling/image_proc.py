import numpy as np
import numpy.random as nr
try:
    from skimage.transform import rotate
except:
    print 'warning: skimage not installed, disable rotation'


def rand_rotate(im, rotate_amp):
    deg = (2 * nr.rand() - 1) * rotate_amp
    # print deg
    rot_im = rotate(im, deg, mode='nearest') * 255
    return rot_im.astype(np.uint8)


def is_color(im):
    if len(im.shape) == 3:
        if im.shape[-1] == 3:
            return True
        else:
            return False
    else:
        return False


def imshow(im):
    im = im.astype(np.uint8)
    if not is_color(im):
        im = im.reshape((im.shape[0], im.shape[1], 1))
        im = np.tile(im, (1, 1, 3))
    else:
        im = im[:, :, ::-1]  # switch to RGB
    #pl.imshow(im)
    #pl.show()


def deprocess(data_pack, mean):
    crop_size = data_pack.shape[-1]
    mean_size = mean.shape[0]
    isclr = is_color(mean)
    imshow(mean)
    if mean_size > crop_size:
        x = (mean_size - crop_size) / 2
        if isclr:
            mean = mean[x:x+crop_size, x:x+crop_size, :]
        else:
            mean = mean[x:x+crop_size, x:x+crop_size]
    imshow(mean)
    batchsize = data_pack.shape[0]
    for i in xrange(batchsize):
        elem = data_pack[i, :, :, :]
        elem = elem.transpose((1, 2, 0)) + mean
        elem[elem > 255] = 255
        elem[elem < 0] = 0
        imshow(elem.astype(np.uint8))


def undo_trans(data, mean):
    data += mean
    data = data.transpose((1, 2, 0))
    if data.shape[-1] == 1:
        data = np.tile(data, (1, 1, 3))
    else:
        data = data[:, :, ::-1]
    print '%0.2f %0.2f' % (data.max(), data.min())
    return data.astype(np.uint8)


class Transformer:
    def __init__(self, crop_size, num_channels, mean_=None, is_train=False, rotate_amp=None):
        self._crop_size = crop_size
        self._in_size = 256
        self._boarder_size = self._in_size - self._crop_size
        self._num_channels = num_channels
        self._is_train = is_train
        self._rotate_amp = rotate_amp
        if self._num_channels > 1 and self._rotate_amp > 0:
            raise Exception("can not rotate color image")
        if type(mean_) == str:
            mean_mat = np.load(mean_)
            self._mean = mean_mat.mean(axis=-1).mean(axis=-1).reshape(1, 3, 1, 1)  # mean value
        else:
            self._mean = mean_

    # @profile
    def transform(self, im):
        if len(im.shape) == 1:
            im = im.reshape((self._in_size, self._in_size)) if (self._num_channels == 1) else \
                im.reshape((self._num_channels, self._in_size, self._in_size))
        # rotation
        im1 = rand_rotate(im, self._rotate_amp) if self._rotate_amp is not None else im

        # translation and flip
        if len(im1.shape) == 2:  # gray scale
            im1 = im1.reshape((1, im1.shape[0], im1.shape[1]))
        x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
        y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2

        if nr.random() > 0.5 and self._is_train:
            im2 = im1[:, y:y+self._crop_size, x+self._crop_size:x:-1]
        else:
            im2 = im1[:, y:y+self._crop_size, x:x+self._crop_size]
        return im2

    def transform_all(self, imlist):
        processed = []
        for im in imlist:
            if im.shape[-1] == self._crop_size:
                processed.append(im.reshape(1, self._num_channels, self._crop_size, self._crop_size))
                continue
            # translation and flip for image
            x = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2
            y = nr.randint(self._boarder_size) if self._is_train else self._boarder_size / 2

            if nr.random() > 0.5 and self._is_train:
                trans_image = im[:, y:y+self._crop_size, x+self._crop_size:x:-1]
            else:
                trans_image = im[:, y:y+self._crop_size, x:x+self._crop_size]
            processed.append(trans_image.reshape(1, self._num_channels, self._crop_size, self._crop_size))
        # data = np.concatenate(processed, axis=0)
        data = np.reshape(processed, (len(imlist), self._num_channels, self._crop_size, self._crop_size))
        return data - self._mean