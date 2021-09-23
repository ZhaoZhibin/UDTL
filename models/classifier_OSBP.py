from torch import nn
import numpy as np


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class ClassifierNet(nn.Module):
    def __init__(self, in_feature,output_num, max_iter=10000.0, trade_off_adversarial='Step', lam_adversarial=1.0):
        super(ClassifierNet, self).__init__()
        self.fc = nn.Linear(in_feature, output_num)

        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial
        self.__in_features = 1

    def forward(self, x, adaption=False):
        if self.training:
            self.iter_num += 1
        if self.trade_off_adversarial == 'Cons':
            coeff = self.lam_adversarial
        elif self.trade_off_adversarial == 'Step':
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        else:
            raise Exception("loss not implement")
        x = x * 1.0
        if adaption == True:
            x.register_hook(grl_hook(coeff))
        y = self.fc(x)
        return y

    def output_num(self):
        return self.__in_features
