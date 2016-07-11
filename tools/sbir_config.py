from collections import namedtuple

def load_model_config(dataset=''):
    config = {
        'verbose': False,
        'gpu_id': 0,
        'input_dim': 256,
        'crop_dim': 225,
        'mean_val': 250.42, # mean value
        'batchsize': 128,  # batchsize for feature extraction
        'feat_layer':'norm7_sketch',
        'dataset': dataset,
        'DATASET_ROOT': 'data/sbir_cvpr2016',
        'deploy_file': 'data/models/sketch_fc7_norm_deploy.prototxt',
        'FEAT_PATH': 'data/feats/%s_feats_val.mat' % dataset,
        'DB_PATH': 'data/dbs/%s/%s_edge_db_val.mat' % (dataset, dataset)
    }
    return namedtuple('Struct', config.keys())(*config.values())


if __name__ == "__main__":
    config = load_model_config()
