import numpy as np
import matplotlib.pyplot as plt
# bs = [10, 15, 20, 25, 30, 35, 40, 45, 50]
lr = [0.02, 0.1, 0.5]
rep = 5
prefix = 'res/1d/res'

N = len(lr) * rep
test_error = np.zeros((N))
gradW = np.zeros((N))
gradx = np.zeros((N))
#batch_size = np.zeros((N))
learning_rate = np.zeros((N))
train_X, train_y = [], []
test_X, test_y = [], []

i = 0
j = 0
for l in lr:
    for r in range(rep):
        res = np.load(prefix+'_lr'+str(j+1)+'_'+str(r+1)+'.npz')
        test_error[i] = res['test_loss']
        gradW[i] = res['gradW']
        gradx[i] = res['gradx']
        #batch_size[i] = b
        learning_rate[i] = l
        
        train_X.append(res['train_X'])
        train_y.append(res['train_y'])
        test_X.append(res['test_X'])
        test_y.append(res['test_y'])
        i += 1
    j += 1

ax = plt.scatter(gradW, gradx, c=learning_rate)
cb = plt.colorbar(ax)
cb.set_label('learning rate', fontsize=16)
plt.xlabel(r'$g_w$', fontsize=16)
plt.ylabel(r'$g_x$', fontsize=16)
# plt.savefig('figs/cifar_bs.png', bbox_inches='tight')
plt.show()

# plt.plot(gradx, test_error, '.')
# plt.show()

# plt.plot(gradW, test_error, '.')
# plt.show()