import numpy as np
import os

# data = np.load('graph_save/02/00161.npz')
# data = np.load('graph_save/umn/test01/00161.npz')

# data = np.load('graph_save/hajj/01/00062.npz')
# data = np.load('graph_cross/lp/01/00161.npz')
# print('data : node feature size (node count, feature dim) :', data['f'].shape)

# print('data : adj2 size : ', data['a2'].shape)

# data = np.load(00062.npz')
folder = 'graph_save2/hajj/01/'
a = []
a2 = []
for cur_file in os.listdir(folder):
    filename = os.path.join(folder, cur_file)
    data = np.load(filename)
    # adj, features, adj2 = data['a'], data['f'], data['a2']
    a.append(np.std([i for i in data['a'].flatten() if i != 0]))
    a2.append(np.std([i for i in data['a2'].flatten() if i != 0]))
print(np.mean(a), np.mean(a2), np.mean(a2) / np.mean(a))

# data = np.load('graph_cross/lp/01/00161.npz')
# print('data : node feature size (node count, feature dim) :', data['f'].shape)
#
# print('data : adj2 size : ', data['a2'].shape)

data = np.load('graph_save3/lp/01/00065.npz')
print(np.std([i for i in data['a'].flatten() if i != 0]))
print(np.std([i for i in data['a2'].flatten() if i != 0]))
# data = np.load('graph_cross/lp/01/00161.npz')
# print('data : node feature size (node count, feature dim) :', data['f'].shape)
#
# print('data : adj2 size : ', data['a2'].shape)

data = np.load('graph_save3/lp/01/00068.npz')
# data = np.load('graph_cross/lp/01/00161.npz')
# print('data : node feature size (node count, feature dim) :', data['f'].shape)
# print('data : adj2 size : ', data['a2'].shape)

print(np.std([i for i in data['a'].flatten() if i != 0]))
print(np.std([i for i in data['a2'].flatten() if i != 0]))
# print('total sum:', len(np.where(data['a2'] >= 0)[0]))
# print('zero sum:', len(np.where(data['a2'] == 0)[0]))
# print('one sum:', len(np.where(data['a2'] == 1)[0]))
# print(data['a2'])
# print(np.where(data['a2'][98] != 0))
# print('-' * 20)
# print('data : adj size : ', data['a'].shape)
# print('total sum:', len(np.where(data['a'] >= 0)[0]))
# print('zero sum:', len(np.where(data['a'] == 0)[0]))
# print('one sum:', len(np.where(data['a'] == 1)[0]))
# print(data['a'])
# print('-' * 20)
# test = data['a'] * data['a2']
# print('data : test size : ', test.shape)
# print('total sum:', len(np.where(test >= 0)[0]))
# print('zero sum:', len(np.where(test == 0)[0]))
# print('one sum:', len(np.where(test == 1)[0]))
# print(test)

# for i in range(data['a'].shape[0]):
#     for j in range(data['a'].shape[1]):
#         if data['a'][i, j] != 0 or True:
#             print('%.10f' % data['a'][i, j], end=' ')
#     print()
#     break
