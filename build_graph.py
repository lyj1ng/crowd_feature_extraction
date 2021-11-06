import numpy as np
from utils import cosine_similarity
import time

# with open('F:/videos/test_tb.csv', 'r') as fp:
with open('graph.csv', 'r') as fp:
    nodes = fp.readlines()
    node = nodes[-1]
    node = node.strip().split(',')
    attribute, position, vpoint, vwave, frame_idx = node
    upper_bound = int(frame_idx)+1
for idx in range(1,upper_bound):
    with open('graph.csv', 'r') as fp:
        nodes = fp.readlines()
    nodes_info = []
    for node in nodes[1:]:
        node = node.strip().split(',')
        attribute, position, vpoint, vwave ,frame_idx = node
        if int(frame_idx) != idx:
            continue
        attribute = float(attribute)
        position = np.array([float(i) for i in position.split(';')])
        vpoint = [float(i) for i in vpoint.split(';')]
        vwave = [float(i) for i in vwave.split(';')]
        nodes_info.append([position, vpoint, vwave, attribute])
        # print(position)
        # break

    start_time = time.time()
    graph = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes_info)):
        for j in range(i + 1, len(nodes_info)):
            n1 = nodes_info[i]
            n2 = nodes_info[j]
            vp_dis = cosine_similarity(n1[1], n2[1])
            v1 = cosine_similarity(n1[1], n2[0] - n1[0])
            v2 = cosine_similarity(n2[1], n1[0] - n2[0])
            graph[i, j] = vp_dis - v1 * v2

    # np.set_printoptions(threshold=np.inf)
    # print(graph)
    sep_time = time.time() - start_time
    print('time consumption : ', sep_time, len(nodes_info))
    for i in range(len(nodes_info)):
        for j in range(len(nodes_info)):
            # print(round(graph[i, j], 1), end='  ')
            pass
        # print()
        pass
