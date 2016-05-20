from scipy.io import loadmat
from time import time


class MemoryBlockManager:
    def __init__(self, data_path, has_label=False):
        self.verbose = False
        self.mean = None
        self.data_path = data_path
        self.num_samples = 0
        self.sample_id = 0
        self.batch_data = None
        self.batch_label = None
        self.has_label = has_label
        self.load_data_block()

    def pop_sample(self):
        if self.sample_id >= self.num_samples:
            self.sample_id = 0
        self.sample_id = self.sample_id + 1
        if self.has_label:
            return self.batch_data[self.sample_id-1, ::], \
                   self.batch_label[self.sample_id-1]
        else:
            return self.batch_data[self.sample_id-1, ::]

    def get_sample(self, sample_idx):
        if self.has_label:
            return self.batch_data[sample_idx, ::], self.batch_label[sample_idx]
        else:
            return self.batch_data[sample_idx, ::]

    def pop_batch(self, batch_size):
        if self.sample_id >= self.num_samples:
            return None
        prev_sample_id = self.sample_id
        self.sample_id = min(self.num_samples, self.sample_id+batch_size)
        data = self.batch_data[prev_sample_id: self.sample_id, ::]
        if self.has_label:
            labels = self.batch_label[prev_sample_id: self.sample_id]
            return data, labels
        else:
            return data, []

    def eof(self):
        return self.sample_id >= self.num_samples

    def pop_batch_circular(self, this_batch_size):
        sample_ind = self.pop_batch_inds_circular(this_batch_size)
        data = self.batch_data[sample_ind, ::]
        if self.has_label:
            labels = self.batch_label[sample_ind]
            return sample_ind, data, labels
        else:
            return sample_ind, data

    def pop_batch_inds_circular(self, this_batch_size):
        sample_ind = range(self.sample_id, self.sample_id+this_batch_size)
        sample_ind = [id % self.num_samples for id in sample_ind]
        self.sample_id = (self.sample_id + this_batch_size) % self.num_samples
        return sample_ind

    def load_data_block(self):
        print 'loading data %s' % self.data_path
        t = time()
        dic = loadmat(self.data_path)
        self.num_samples = dic['data'].shape[0]
        self.batch_data = dic['data']
        if self.has_label:
            self.batch_label = dic['labels']
        print 'data loaded %s (%0.2f sec.)' % (
            self.get_data_shape_str(self.batch_data), time()-t)

    def get_data_shape_str(self, data):
        s = ''
        shape = data.shape
        for d in shape:
            s += '%dx' % d
        s = s[:-1]
        return s

    def get_num_instances(self):
        return self.num_samples