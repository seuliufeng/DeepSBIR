import json
import os


def read_json(fpath):
    with open(fpath) as data_file:
        data = json.load(data_file)
    return data


class SMTSApi(object):
    def __init__(self, ann_path=None, dataset_root=None, name=None):
        self._dataset_root = dataset_root
        self._dbname = name
        if ann_path is None:
            ann_path = os.path.join(dataset_root, name, 'annotation',
                                    '%s_annotation.json' % name)
        self._annotation = read_json(ann_path)

    @property
    def name(self):
        return self._dbname

    def get_triplets(self, target_set='train'):
        if self._annotation is None:
            raise Exception('annotations not loaded')
        if target_set.lower() == 'train':
            return self._annotation['train']['triplets']
        elif target_set.lower() == 'test':
            return self._annotation['test']['triplets']
        else:
            raise Exception('unknown subset: should be train or test')

    def get_images(self, target_set='train'):
        if target_set.lower() == 'train':
            return self._annotation['train']['images']
        elif target_set.lower() == 'test':
            return self._annotation['test']['images']
        else:
            raise Exception('unknown subset: should be train or test')

    def get_image_pathes(self, image_inds, target_set):
        images = self.get_images(target_set)
        im_root = os.path.join(self._dataset_root, self._dbname,
                               target_set, 'images')
        impathes = []
        for image_id in image_inds:
            impathes.append(os.path.join(im_root, images[image_id]))
        return impathes

    def get_image_path(self, image_id, target_set):
        if self._dataset_root is None:
            raise Exception('should pass dataset root in initialization')
        im_name = self._annotation[target_set]['images'][image_id]
        return os.path.join(self._dataset_root, '%s/%s/%s' % (target_set, 'images', im_name))

    def get_sketch_path(self, sketh_id, target_set):
        if self._dataset_root is None:
            raise Exception('should pass dataset root in initialization')
        sk_name = self._annotation[target_set]['images'][sketh_id]
        return os.path.join(self._dataset_root, '%s/%s/%s' % (target_set, 'sketches', sk_name))


if __name__ == "__main__":
    afile = '/Users/liufeng/Projects/sbir_release/sbir/data/sbir_cvpr2016/shoes/annotation/shoes_annotation.json'
    smts_api = SMTSApi(afile)
    train_triplets = smts_api.get_triplets('train')
    test_images = smts_api.get_images('test')
    train_images = smts_api.get_images('train')
    test_triplets = smts_api.get_triplets('test')
    test_images = smts_api.get_images('test')