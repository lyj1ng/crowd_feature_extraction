import cv2
import numpy as np
import time
from utils import *

from utils import cosine_similarity, constrain_max_velocity

print('\ncv2 status : ', cv2.useOptimized())

# folder = 'D:\\simulation\\loveparade\\dataset\\scene1\\training\\03\\'
# save_folder = 'graph_save\\03\\'

folder = 'D:\\simulation\\loveparade\\dataset\\scene1\\testing\\02\\'
save_folder = 'graph_save\\test02\\'

# folder = 'C:\\Users\\forev\\Documents\\PycharmProjects\\MNAD-master\\dataset\\umn\\training\\frames\\02\\'
# save_folder = 'graph_save\\umn\\02\\'

# folder = 'C:\\Users\\forev\\Documents\\PycharmProjects\\MNAD-master\\dataset\\umn\\testing\\frames\\02\\'
# save_folder = 'graph_save\\umn\\test02\\'

# cap = cv2.VideoCapture('./stopandgo.mp4')
frame_index = 0
# 获取第一帧
frame1 = cv2.imread(folder + '0' * (5 - len(str(frame_index))) + str(frame_index) + ".jpg")
frame_index += 1

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
print(hsv.dtype)
print(frame1.shape)
# 遍历每一行的第1列
hsv[..., 1] = 255
width, height = frame1.shape[1], frame1.shape[0]
width, height = int(width), int(height)
# height,width = int(width), int(height)
print('video size  : ', width, height)
with open('./instability_graph_from_video.csv', 'w+') as fp:
    # 进行节点稳定性信息的保存
    fp.write('frame index,node position,ent,mutual info:up,mutual info:down,mutual info:left,'
             'mutual info:right\n')
# 初始化
# 半径设置 2022-04-19 lp:25-25
#                   umn: 10-10
render_radius = 25  #
cal_radius = 25
axis_i = list(range(cal_radius, height - cal_radius, cal_radius * 2))
axis_j = list(range(cal_radius, width - cal_radius, cal_radius * 2))
# 所有点位的x，y轴位置
posis = []  # 存储所有node的position
position_map = {}  # position_map[position] = its_index
node_index = 0
for i in axis_i:
    for j in axis_j:
        # 遍历两个方向轴，得到所有点位的坐标
        posis.append((i, j))
        position_map[str(i) + ';' + str(j)] = node_index
        node_index += 1

# frame_idx = 0
vary = None
index = -1
visualize = False
oldFasion = False
while True:
    try:
        frame2 = cv2.imread(folder + '0' * (5 - len(str(frame_index))) + str(frame_index) + ".jpg")
        frame_index += 1

        node_feature = [[] for _ in range(node_index)]
        adj = [[0 for ii in range(node_index)] for jj in range(node_index)]

    except:
        print('\n\'\'\'End of File\'\'\'\n')
        break

    index += 1
    if index % 3 != 0:  # 每 N 帧计算一次瞬时速度
        continue
    start_time = time.time()
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 7, 1.5, 0)
    # prev 输入8位单通道图片  next 下一帧图片，格式与prev相同  flow 与返回值相同，得到一个CV_32FC2格式的光流图，与prev大小相同
    # pyr_scale 构建图像金字塔尺度 levels 图像金字塔层数  winsize 窗口尺寸，值越大探测高速运动的物体越容易，但是越模糊，同时对噪声的容错性越强
    # iterations 对每层金字塔的迭代次数  poly_n 每个像素中找到多项式展开的邻域像素的大小。越大越光滑，也越稳定
    # poly_sigma 高斯标准差，用来平滑倒数   flags 光流方式，如FARNEBACK_GAUSSIAN

    # 笛卡尔坐标转换为极坐标，获得极轴和极角
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # print(index)
    if True:
        frame = rgb

        filter_kernel = 15
        frame = cv2.blur(frame, (filter_kernel, filter_kernel))  # 对画面进行 均值滤波
        # 一些参数：渲染半径 与 速度计算半径

        if not vary:  # 第一次进行vary的初始化
            vary = [[] for _ in range(len(posis))]
        # 对于所有的node 进行局部速度计算
        #  此处需要优化：对每个位置遍历复杂度太高了
        #  应该对每个rgb遍历，分配给每个位置，最后对每个位置求 weighted average
        plots = []  # 存储所有node的即时local color
        for posi in posis:
            # print(posi,index)
            c = local_color_from_render((posi[0], posi[1]), frame, cal_radius, graph_sample=False)
            c = [int(cc) if cc > 0 else 0 for cc in c]
            plots.append(c)

        # 局部速度可视化 同时更新 vary：存储过去一段时间的速度变化，用于计算信息熵
        for i in range(len(posis)):
            # print(i)
            posi = posis[i]
            hue, sat, val = rgb_to_hsv(plots[i][1], plots[i][2], plots[i][0])
            hue = int(hue // 30)  # 速度方向分箱 的 信息熵 ：也可以计算速度大小分箱的信息熵 即val
            if len(vary[i]) < 20:  # 如果速度变化不足20个，则先不计算信息熵
                vary[i].append(hue)
            else:
                # 数组的拼接
                vary[i] = vary[i][1:] + [hue]
                # 计算信息熵
                ent = calc_ent(np.array(vary[i]))
                # 计算互信息：上下左右
                mutual_info = []

                this_position = [str(p) for p in np.array(posi)]
                this_position = ';'.join(this_position)
                this_index = position_map[this_position]
                for neighbor in [[0, -2 * cal_radius], [0, 2 * cal_radius], [-2 * cal_radius, 0],
                                 [2 * cal_radius, 0]]:
                    neighbor_position = np.array(posi) + np.array(neighbor)
                    neighbor_position = [str(p) for p in neighbor_position]
                    neighbor_position = ';'.join(neighbor_position)
                    # print(neighbor_position,position_map)
                    if neighbor_position in position_map:
                        n_vary = np.array(vary[position_map[neighbor_position]])
                        mi = calc_ent_grap(np.array(vary[i]), n_vary)
                        adj[this_index][position_map[neighbor_position]] = mi
                        if oldFasion:
                            mutual_info.append(mi)
                    elif oldFasion:
                        mutual_info.append(None)
                # 存这个点位的信息
                ent_vec = [0] * 15
                if ent < 1:
                    # 0到1之间的ent，占据前8个vec
                    ent_vec[int(ent*8)] = 1
                elif ent < 3:
                    # 1-3之间的ent，占据后5个vec
                    ent_vec[int(3*(ent-1))] = 1
                # 最后两个记录大ent
                elif ent < 4:
                    ent_vec[-2] = 1
                else:
                    ent_vec[-1] = 1

                node_feature[this_index] = ent_vec[:]
                if oldFasion:
                    with open('./instability_graph_from_video.csv', 'a+') as fp:
                        fp.write(
                            '0' * (5 - len(str(frame_index))) + str(frame_index) + ',' + str(posi[0]) + ';' + str(posi[1])
                            + ',' + str(ent) + ',' + str(mutual_info) + '\n')
            if visualize:
                cv2.circle(frame, (posi[1], posi[0]), render_radius, plots[i], -1)

        if node_feature[0]:
            adj = np.array(adj)
            node_feature = np.array(node_feature)
            # print(adj)
            # print(node_feature)
            # input()
            # np.save(save_folder+'0' * (5 - len(str(frame_index))) + str(frame_index)+'.npy', adj)
            np.savez(save_folder + '0' * (5 - len(str(frame_index))) + str(frame_index) + '.npz', f=node_feature, a=adj)

        # real-time frame fresh rate
        sep_time = time.time() - start_time
        fr = round(1 / sep_time, 1) if sep_time else 'inf'
        print('\rframe_rate : ', fr, end=' fps ')

        # frame_idx += 1
        if visualize:
            cv2.namedWindow('lk_track', 0)
            cv2.imshow('lk_track', frame)
            ch = cv2.waitKey(3)
            if ch == ord(' '):  # quit
                break

    sep_time = time.time() - start_time
    print('\rframe_rate : ', round(1 / sep_time, 1), end=' fps ' + '.' * (index // 3 % 4) + ' ' * 10)

    prvs = next_frame

if visualize:
    cv2.destroyAllWindows()
