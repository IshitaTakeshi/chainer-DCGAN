import os
from os.path import join
import math
import pickle

import pylab  # TODO use matplotlib.pyplot

import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L

from tqdm import tqdm
from dataset import load_images

image_shape = (96, 96)

image_dir = join(os.getenv("HOME"), ".julia", "v0.4", "ClothingRecommender",
                 "dataset", "amebafurugiya", "data", "images", "full")
out_image_dir = './out_images'
out_model_dir = './out_models'


nz = 100          # # of dim for Z
batchsize = 100
n_epoch = 10000
n_train = 200000
image_save_interval = 50000

# read all images

@profile
def generate_data(images):
    data = np.zeros((batchsize, 3, *image_shape), dtype=np.float32)
    for j in range(batchsize):
        try:
            rnd = np.random.randint(0, len(images))
            rnd2 = np.random.randint(2)
            img = images[rnd]
            if rnd2 == 0:
                data[j, :, :, :] = (img[:, :, ::-1] - 128.) / 128.
            else:
                data[j, :, :, :] = (img[:, :, :] - 128.) / 128.
        except Exception as e:
            print(e)

    print("loaded")
    data = cuda.to_gpu(data)
    print("moved")
    data = Variable(data)
    print("converted")
    return data


class ELU(function.Function):

    """Exponential Linear Unit."""
    # https://github.com/muupan/chainer-elu

    def __init__(self, alpha=1.0):
        self.alpha = np.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == np.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (np.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * np.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,


def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)


class Generator(chainer.Chain):

    def __init__(self):
        super(Generator, self).__init__(
            l0z=L.Linear(nz, 6 * 6 * 512, wscale=0.02 * math.sqrt(nz)),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2,
                                  pad=1, wscale=0.02 * math.sqrt(4 * 4 * 512)),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2,
                                  pad=1, wscale=0.02 * math.sqrt(4 * 4 * 256)),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2,
                                  pad=1, wscale=0.02 * math.sqrt(4 * 4 * 128)),
            dc4=L.Deconvolution2D(64, 3, 4, stride=2, pad=1,
                                  wscale=0.02 * math.sqrt(4 * 4 * 64)),
            bn0l=L.BatchNormalization(6 * 6 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)),
                      (z.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x


class Discriminator(chainer.Chain):

    def __init__(self):
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(3, 64, 4, stride=2, pad=1,
                               wscale=0.02 * math.sqrt(4 * 4 * 3)),
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1,
                               wscale=0.02 * math.sqrt(4 * 4 * 64)),
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1,
                               wscale=0.02 * math.sqrt(4 * 4 * 128)),
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1,
                               wscale=0.02 * math.sqrt(4 * 4 * 256)),
            l4l=L.Linear(6 * 6 * 512, 2, wscale=0.02 * math.sqrt(6 * 6 * 512)),
            bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        # no bn because images from generator will katayotteru?
        h = elu(self.c0(x))
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))
        h = elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l


def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))


@profile
def train_dcgan_labeled(images, gen, dis):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (xp.random.uniform(-1, 1, (100, nz), dtype=np.float32))

    zeros = Variable(xp.zeros(batchsize, dtype=np.int32))
    ones = Variable(xp.ones(batchsize, dtype=np.int32))
    for epoch in tqdm(range(n_epoch)):
        perm = np.random.permutation(n_train)

        for i in range(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise

            # print "load image start ", i
            # print "load image done"

            # train generator
            z = xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32)
            z = Variable(z)
            x = gen(z)
            yl = dis(x)
            L_gen = F.softmax_cross_entropy(yl, zeros)
            L_dis = F.softmax_cross_entropy(yl, ones)

            # train discriminator

            x = generate_data(images)
            yl2 = dis(x)
            L_dis += F.softmax_cross_entropy(yl2, zeros)

            # print "forward done"

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()

            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()

            # print "backward done"

            if i % image_save_interval == 0:
                pylab.rcParams['figure.figsize'] = (16.0, 16.0)
                pylab.clf()
                vissize = 100
                z = zvis
                z[50:, :] = (
                    xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x = x.data.get()
                for i_ in range(100):
                    tmp = ((np.vectorize(clip_img)(
                        x[i_, :, :, :]) + 1) / 2).transpose(1, 2, 0)
                    pylab.subplot(10, 10, i_ + 1)
                    pylab.imshow(tmp)
                    pylab.axis('off')
                pylab.savefig('%s/vis_%d_%d.png' % (out_image_dir, epoch, i))

        serializers.save_hdf5("%s/dcgan_model_dis_%d.h5" %
                              (out_model_dir, epoch), dis)
        serializers.save_hdf5("%s/dcgan_model_gen_%d.h5" %
                              (out_model_dir, epoch), gen)
        serializers.save_hdf5("%s/dcgan_state_dis_%d.h5" %
                              (out_model_dir, epoch), o_dis)
        serializers.save_hdf5("%s/dcgan_state_gen_%d.h5" %
                              (out_model_dir, epoch), o_gen)
        exit(0)


xp = cuda.cupy
cuda.get_device(0).use()

gen = Generator()
dis = Discriminator()
gen.to_gpu()
dis.to_gpu()


try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
except:
    pass

images = load_images(image_dir, image_shape)
train_dcgan_labeled(images, gen, dis)
