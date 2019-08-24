import argparse
import torch

from config import Config
from builder import ModelBuilder

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--func', type=str, default='train')
parser.add_argument('-m', '--model', type=str, default='RNN')
parser.add_argument('-s', '--splitting', type=int, default=3)


def main(conf, is_train=True, model_name='RNN', pre=None):
    havecuda = torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    if havecuda:
        torch.cuda.manual_seed(conf.seed)

    model = ModelBuilder(havecuda, conf, model_name)
    if is_train:
        model.train(pre)
    else:
        model.eval(pre)


if __name__ == '__main__':
    A = parser.parse_args()
    if A.splitting == 1:
        conf = Config(11, 1, A.model)
        if A.func == 'train':
            main(conf, True, A.model, 'lin')
        elif A.func == 'eval':
            main(conf, False, A.model, 'lin')
        else:
            raise Exception('wrong func')
    if A.splitting == 2:
        conf = Config(11, 2, A.model)
        if A.func == 'train':
            main(conf, True, A.model, 'ji')
        elif A.func == 'eval':
            main(conf, False, A.model, 'ji')
        else:
            raise Exception('wrong func')
    if A.splitting == 3:
        conf = Config(2, 3, A.model)
        if A.func == 'train':
            main(conf, True, A.model, 'four')
        elif A.func == 'eval':
            main(conf, False, A.model, 'four')
        else:
            raise Exception('wrong func')
