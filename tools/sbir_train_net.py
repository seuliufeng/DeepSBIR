from train import train_net
import sys
import caffe
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Deep SBIR')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--output', dest='output_dir',
                        help='dir to save the model',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--snapstep', dest='snapshot_iters',
                        help='snapshot every snapstep iters',
                        default=500, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    train_net(args.solver, args.output_dir, args.pretrained_model, max_iters=args.max_iters, snapshot_iters=args.snapshot_iters)
