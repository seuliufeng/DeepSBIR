import yaml
import caffe
from multiprocessing import Process, Queue
from sbir_util.batch_manager import MemoryBlockManager
from image_proc import Transformer
from sample_util import *


class TripletSamplingLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TripletSamplingLayer."""
        layer_params = yaml.load(self.param_str)
        self.mini_batchsize = layer_params['batch_size']
        self.layer_params = layer_params
        # anchor
        top[0].reshape(self.mini_batchsize, 1, 225, 225)
        # pos
        top[1].reshape(self.mini_batchsize, 1, 225, 225)
        # neg
        top[2].reshape(self.mini_batchsize, 1, 225, 225)
        # weights blob: dummy, we don't need that, just fill it with 1.
        top[3].reshape(self.mini_batchsize, 1)

    def create_sample_fetcher(self):
        self._blob_queue = Queue(10)
        self._prefetch_process = TripletSamplingDataFetcher(self._blob_queue, self.layer_params)
        self._prefetch_process.start()
        def cleanup():
            print 'Terminating BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join()
        import atexit
        atexit.register(cleanup)

    def forward(self, bottom, top):
        blobs = self._blob_queue.get()
        top[0].data[...] = blobs[0].astype(np.float32, copy=False)
        top[1].data[...] = blobs[1].astype(np.float32, copy=False)
        top[2].data[...] = blobs[2].astype(np.float32, copy=False)
        top[3].data[...] = 1.  # sample weights, we don't need that

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Donot reshape once started."""
        pass


class TripletSamplingDataFetcher(Process):
    def __init__(self, queue, layer_params):
        """Setup the TripletSamplingDataLayer."""
        super(TripletSamplingDataFetcher, self).__init__()
        self._queue = queue
        mean = layer_params['mean']
        self._phase = layer_params['phase']
        self.sketch_transformer = Transformer(225, 1, mean, self._phase == "TRAIN")
        self.anc_bm = MemoryBlockManager(layer_params['sketch_dir'])
        self.pos_neg_bm = MemoryBlockManager(layer_params['image_dir'])
        self.hard_ratio = layer_params['hard_ratio']
        self.mini_batchsize = layer_params['batch_size']
        self.load_triplets(layer_params['triplet_path'])

    def load_triplets(self, triplet_path):
        self.triplets, self.neg_list = load_triplets(triplet_path, self._phase)

    def get_next_batch(self):
        anc_batch = []; pos_batch = []; neg_batch = []
        # sampling
        anc_inds = self.anc_bm.pop_batch_inds_circular(self.mini_batchsize)
        pos_inds, neg_inds = sample_triplets(anc_inds, self.triplets, self.neg_list, self.hard_ratio)
        # fetch data
        for (anc_id, pos_id, neg_id) in zip(anc_inds, pos_inds, neg_inds):
            anc_batch.append(self.anc_bm.get_sample(anc_id).reshape((1, 256, 256)))
            pos_batch.append(self.pos_neg_bm.get_sample(pos_id).reshape((1, 256, 256)))
            neg_batch.append(self.pos_neg_bm.get_sample(neg_id).reshape((1, 256, 256)))
        # apply transform
        anc_batch = self.sketch_transformer.transform_all(anc_batch)
        pos_batch = self.sketch_transformer.transform_all(pos_batch)
        neg_batch = self.sketch_transformer.transform_all(neg_batch)
        self._queue.put((anc_batch, pos_batch, neg_batch))

    def run(self):
        print 'TripletSamplingDataFetcher started'
        while True:
            self.get_next_batch()

