import os
import time
import numpy as np
import argparse
import json
import torch

from src.utils import load_net, load_data, \
                      get_sharpness, get_nonuniformity, \
                      eval_accuracy, eval_output,\
                      get_gradW, get_gradx

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--gpuid',default='0,')
    argparser.add_argument('--dataset',default='fashionmnist',
                            help='dataset choosed, [fashionmnist] | cifar10, 1dfunction')
    argparser.add_argument('--network', default='vgg')
    argparser.add_argument('--num_classes', type=int, default=2)
    argparser.add_argument('--n_samples_per_class', type=int,
                           default=500, help='training set size, [1000]')
    argparser.add_argument('--batch_size', type=int,
                            default=1000, help='batch size')
    argparser.add_argument('--k', type=int, default=1)
    argparser.add_argument('--model_file', default='fnn.pkl',
                            help='file name of the pretrained model')
    argparser.add_argument('--res_file', default='fnn_res.npz')
    args = argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

    print('===> Config:')
    print(json.dumps(vars(args),indent=2))
    return args


def main():
    args = get_args()

    # load model
    criterion = torch.nn.MSELoss().cuda()
    train_loader,test_loader = load_data(args.dataset,
                                         args.num_classes,
                                         train_per_class=
                                             args.n_samples_per_class,
                                         batch_size=1)
    net = load_net(args.network, args.dataset, args.num_classes)
    net.load_state_dict(torch.load('res/'+args.model_file))

    # Evaluate models
    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)

    print('===> Basic information of the given model: ')
    print('\t train loss: %.2e, acc: %.2f'%(train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f'%(test_loss, test_accuracy))
    
    print('===> Calculating output: ')
    train_X, train_y = eval_output(net, train_loader)
    test_X, test_y = eval_output(net, test_loader)
    
    print('===> Compute gradient W:')
    gradW = get_gradW(net, train_loader, ndata=args.num_classes*args.n_samples_per_class, k=args.k)
    print('gradient W is %.3e\n' % (gradW))
    
    print('===> Compute gradient x:')
    gradx = get_gradx(net, train_loader, ndata=args.num_classes*args.n_samples_per_class, k=args.k)
    print('gradient x is %.3e\n' % (gradx))

    # print('===> Compute sharpness:')
    # sharpness = get_sharpness(net, criterion, train_loader, \
    #                             n_iters=10, verbose=True, tol=1e-4)
    # print('Sharpness is %.2e\n'%(sharpness))

    # print('===> Compute non-uniformity:')
    # non_uniformity = get_nonuniformity(net, criterion, train_loader, \
    #                                     n_iters=10, verbose=True, tol=1e-4)
    # print('Non-uniformity is %.2e\n'%(non_uniformity))
    
    np.savez('res/'+args.res_file, train_loss=train_loss, train_acc=train_accuracy, test_loss=test_loss, test_acc=test_accuracy, gradW=gradW, gradx=gradx, train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y)

if __name__ == '__main__':
    main()
