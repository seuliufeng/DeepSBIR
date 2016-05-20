# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

SNAPSHOT_ITERS = 500

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, output_dir, pretrained_model=None, snapshot_iters=500):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.SNAPSHOT_ITERS = snapshot_iters

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            self.load_pretained_model(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].create_sample_fetcher()

    def load_pretained_model(self, pretrained_model):
        models = pretrained_model.split(',')
        for model in models:
            print ('Loading pretrained model '
                   'weights from {:s}').format(model)
            if not os.path.exists(model):
                raise Exception('model %s do not exist' % model)
            self.solver.net.copy_from(model)


    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ''
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)


    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % self.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()


def train_net(solver_prototxt, output_dir,
              pretrained_model=None, max_iters=40000, snapshot_iters=500):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, output_dir,
                       pretrained_model=pretrained_model, snapshot_iters=snapshot_iters)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'

