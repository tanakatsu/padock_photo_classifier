import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):

    def __init__(self, outputSize=1, train=True):
        initializer = chainer.initializers.GlorotNormal()
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 20, 5, initialW=initializer),
            conv2=L.Convolution2D(20, 50, 5, initialW=initializer),
            l1=L.Linear(None, 500, initialW=initializer),
            l2=L.Linear(500, outputSize, initialW=initializer),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        y = self.l2(h)
        return y
