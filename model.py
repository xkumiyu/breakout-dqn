import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl


class QFunction(chainer.Chain):
    insize = (210, 160, 3)

    def __init__(self, n_actions=4):
        w = chainer.initializers.Normal(0.01)
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 32, 8, 8, initialW=w)
            self.conv2 = L.Convolution2D(32, 64, 4, 4, initialW=w)
            self.conv3 = L.Convolution2D(64, 64, 3, 3, initialW=w)
            self.fc = L.Linear(128, n_actions)

    def __call__(self, x):
        h = self.prepare(x)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.fc(h)
        return chainerrl.action_value.DiscreteActionValue(h)

    def prepare(self, x):
        x = x.transpose(0, 3, 1, 2)
        return x
