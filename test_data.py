import numpy as np

data = np.load('graph_save/02/00161.npz')
# data = np.load('graph_save/umn/test01/00161.npz')

# data = np.load('graph_save/hajj/01/00062.npz')
# data = np.load('graph_cross/lp/01/00161.npz')
print('data : node feature size (node count, feature dim) :', data['f'].shape)

print('data : adj2 size : ', data['a2'].shape)

data = np.load('graph_save2/lp/02/00062.npz')
# data = np.load('graph_cross/lp/01/00161.npz')
print('data : node feature size (node count, feature dim) :', data['f'].shape)

print('data : adj2 size : ', data['a2'].shape)

data = np.load('graph_save3/lp/test02/00062.npz')
# data = np.load('graph_cross/lp/01/00161.npz')
print('data : node feature size (node count, feature dim) :', data['f'].shape)

print('data : adj2 size : ', data['a2'].shape)

data = np.load('graph_save/umn/test01/00062.npz')
# data = np.load('graph_cross/lp/01/00161.npz')
print('data : node feature size (node count, feature dim) :', data['f'].shape)

print('data : adj2 size : ', data['a2'].shape)
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
#         if data['a'][i, j] != 0:
#             print(data['a'][i, j], test[i, j])
