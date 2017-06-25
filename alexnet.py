import chainer
import chainer.links as L
import chainer.functions as F


class FromCaffeAlexnet(chainer.Chain):
    insize = 128
    def __init__(self, n_out):
        super(FromCaffeAlexnet, self).__init__(
            # conv1=L.Convolution2D(None, 96, 11, stride=2),
            # conv2=L.Convolution2D(None, 256, 5, pad=2),
            # conv3=L.Convolution2D(None, 384, 3, pad=1),
            # conv4=L.Convolution2D(None, 384, 3, pad=1),
            # conv5=L.Convolution2D(None, 256, 3, pad=1),
            # my_fc6=L.Linear(None, 4096),
            # my_fc7=L.Linear(None, 1024),
            # my_fc8=L.Linear(None, n_out),

            # Don't use None when you copy parameters
            conv1=L.Convolution2D(3, 96, 11, stride=2),
            conv2=L.Convolution2D(96, 256, 5, pad=2),
            conv3=L.Convolution2D(256, 384, 3, pad=1),
            conv4=L.Convolution2D(384, 384, 3, pad=1),
            conv5=L.Convolution2D(384, 256, 3, pad=1),
            # my_fc6=L.Linear(None, 4096),
            # my_fc7=L.Linear(None, 1024),
            # my_fc8=L.Linear(None, n_out),
            my_fc6=L.Linear(256 * 7 * 7, 4096),
            my_fc7=L.Linear(4096, 1024),
            my_fc8=L.Linear(1024, n_out),
        )
        self.train = True
 
    def __call__(self, x):
        # for chainer v1.x.x 
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
        h = self.my_fc8(h)

        # for chainer v2.x.x
        # You don't need to use DelGradient hook.

        # with chainer.no_backprop_mode():
        #     h = F.max_pooling_2d(F.local_response_normalization(
        #         F.relu(self.conv1(x))), 3, stride=2)
        #     h = F.max_pooling_2d(F.local_response_normalization(
        #         F.relu(self.conv2(h))), 3, stride=2)
        #     h = F.relu(self.conv3(h))
        #     h = F.relu(self.conv4(h))
        #     h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        #     with chainer.force_backprop_mode():
        #         h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        #         h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
        #         h = self.my_fc8(h)

        return h
