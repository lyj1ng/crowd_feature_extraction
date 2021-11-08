import numpy as np
from utils import *
import time

with open('graph.csv', 'r') as fp:  # 读取速度传递图的uppper bound
    node = fp.readlines()[-1]
    node = node.strip().split(',')
    attribute, position, vpoint, vwave, frame_idx = node
    upper_bound = int(frame_idx)
    # print(upper_bound)
with open('instability_graph.csv', 'r') as fp:
    node = fp.readlines()[-1]
    node = node.strip().split(',')
    frame_idx = int(node[0])
    # print(frame_idx)
    upper_bound = min(upper_bound, frame_idx)
with open('vector.csv', 'w+') as fp:
    fp.write('frame_idx,vector_instability,,vector_speed,,,,\n')

# 对所有idx进行循环读取
last = []
for idx in range(1, upper_bound + 1):
    start_time = time.time()
    nodes_info = []  # 速度传递节点
    instability_info = []  # 稳定性节点
    with open('graph.csv', 'r') as fp:
        nodes = fp.readlines()
        for node in nodes[1:]:
            node = node.strip().split(',')
            attribute, position, vpoint, vwave, frame_idx = node
            if int(frame_idx) != idx:
                continue
            attribute = float(attribute)
            position = np.array([float(i) for i in position.split(';')])
            vpoint = [float(i) for i in vpoint.split(';')]
            vwave = [float(i) for i in vwave.split(';')]
            nodes_info.append([position, vpoint, vwave, attribute])
    with open('instability_graph.csv', 'r') as fp:
        nodes2 = fp.readlines()
        for node in nodes2[1:]:
            node = node.strip().split(',')
            frame_idx, position, ent, mi_up, mi_down, mi_left, mi_right = node
            if int(frame_idx) != idx:
                continue
            # attribute = float(attribute)
            position = np.array([float(i) for i in position.split(';')])
            ent = float(ent)
            # vwave = [float(i) for i in vwave.split(';')]
            mi_up = float(mi_up[1:]) if 'No' not in mi_up else 'None'
            mi_left = float(mi_left) if 'No' not in mi_left else 'None'
            mi_down = float(mi_down) if 'No' not in mi_down else 'None'
            mi_right = float(mi_right[:-2]) if 'No' not in mi_right else 'None'
            mi = np.mean([i for i in [mi_up, mi_down, mi_left, mi_right] if i != 'None'])
            instability_info.append([position, ent, mi])
    # 以上得到的是这一帧中的全部信息：速度传递信息 nodes_info 与稳定性信息 insta_info

    graph = np.full((len(nodes), len(nodes)), None)  # edge infomation
    for i in range(len(nodes_info)):
        for j in range(i + 1, len(nodes_info)):
            n1 = nodes_info[i]
            n2 = nodes_info[j]
            if euclid_distance(n1[0], n2[0], root=True) > 10 * 5:
                graph[i, j] = None
            else:
                vp_dis = cosine_similarity(n1[1], n2[1])
                v1 = cosine_similarity(n1[1], n2[0] - n1[0])
                v2 = cosine_similarity(n2[1], n1[0] - n2[0])
                graph[i, j] = vp_dis - v1 * v2
                graph[i, j] = round(graph[i, j], 2)
    # 以上是对nodes info的边的计算

    # for i in range(len(nodes_info)):
    #     for j in range(len(nodes_info)):
    #         print(graph[i, j], end='  ')
    #         pass
    #     print()
    #     pass
    # input()

    input_position = (35, 36)
    vector_speed = []
    vector_instability = []
    ents = []
    mis = []
    for insta_info in instability_info:
        position, ent, mi = insta_info
        # print(position,input_position)
        if euclid_distance(position, input_position, root=True) < 9:
            ents.append(ent)
            mis.append(mi)
    vector_instability = [np.mean(ents), np.mean(mis)] if ents else []
    wave_agent = []
    point_agent = []
    ats = []
    for node in nodes_info:
        position, vpoint, vwave, attribute = node
        wave_agent.append([position, vwave])
        point_agent.append([position, [v/255 for v in vpoint]])
        if euclid_distance(position, input_position, root=True) < 10:
            ats.append(attribute)
        else:
            ats.append(attribute / 10)

    vector_speed = local_color_from_render(input_position, np.array(wave_agent), graph_sample=True)+local_color_from_render(input_position, np.array(point_agent), graph_sample=True)+[np.sum(ats)] if nodes_info else last
    last = vector_speed

    vector_speed = [str(round(v, 2)) for v in vector_speed]
    vector_instability = [str(round(v, 2)) for v in vector_instability]

    sep_time = time.time() - start_time
    print('time consumption : ', round(sep_time, 3), len(nodes_info), end='\t')
    print(vector_instability, vector_speed)
    with open('vector.csv', 'a') as fp:
        if vector_instability:
            fp.write(str(idx)+','+','.join(vector_instability)+','+','.join(vector_speed)+'\n')
