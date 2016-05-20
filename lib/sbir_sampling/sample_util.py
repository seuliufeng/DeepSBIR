import numpy as np
import numpy.random as nr
from sbir_util.smts_api import SMTSApi


def sample_triplets(anc_inds, triplets, neg_list, hard_ratio):
    pos_inds = []
    neg_inds = []
    for anc_id in anc_inds:
        tuples = triplets[anc_id]
        idx = nr.randint(len(tuples))
        pos_id, neg_id = tuples[idx]
        pos_inds.append(pos_id)
        if nr.rand() > hard_ratio:  # sample easy
            nidx = nr.randint(neg_list.shape[1])
            neg_id = neg_list[anc_id, nidx]
        neg_inds.append(neg_id)
    return pos_inds, neg_inds


def load_triplets(triplet_path, subset):
    smts_api = SMTSApi(triplet_path)
    triplets = smts_api.get_triplets(subset)
    return triplets, make_negative_list(triplets)


def make_negative_list(triplets):
    tri_mat = np.array(triplets)
    num_images = tri_mat.shape[0]
    all_inds = np.unique(triplets)
    neg_list = []
    for i in xrange(num_images):
        pos_inds = np.union1d(tri_mat[i, :, 0], tri_mat[i, :, 1])
        neg_inds = np.setdiff1d(all_inds, pos_inds).reshape([1, -1])
        neg_list.append(neg_inds)
    return np.concatenate(neg_list).astype(np.int32)
