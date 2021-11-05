#!/usr/bin/env python

import numpy as np
import cv2 as cv
from utils import *
import time
from sim_data_iterator import SimData


def cal_local_velocity(position, agents, radius=10):
    """
    calculate local density and velocity
    :param position: position to measure:[0,0](position in modeling)
    :param radius: radius R of measure:the bigger R,the smoothing area around position
    :param agents: data of all agents:agent[0] is position and agent[1] is velocity
    :return: local density and local velocity
    """
    local_density = 0
    local_color = np.zeros(3)
    for agent in agents:
        weight = np.exp(-1 * euclid_distance(agent[0], position, root=False) / (radius ** 2))
        local_density += weight
        local_color += weight * np.array(agent[1])
    local_color /= local_density
    return local_color


def sample_points_from_render(position, vis, sample_times=20, radius=20):
    ret = []
    vis = np.array(vis)
    for i in range(sample_times):
        xi, yi = np.random.randint(-1 * radius, radius), np.random.randint(-1 * radius, radius)
        sample_position = (position[0] + xi, position[1] + yi)
        # if sample_position[0]>0  # 此处待加入限制
        # print(sample_position)
        ret.append([sample_position, vis[sample_position]])
    # ret = [[position,vis[position]]]
    return ret


def local_color_from_render(position, vis, radius=10):
    # position = (position[1],position[0])
    agents = sample_points_from_render(position, vis, radius=2 * radius, sample_times=radius)
    return cal_local_velocity(position, agents, radius=radius)


class App:
    def __init__(self, video_src):
        self.cam = SimData(video_src)  # 新建一个仿真数据迭代器对象，用于读取仿真数据的渲染画面
        # 对仿真迭代器的参数进行调整
        self.cam.set('size', (10, 10))
        self.cam.set('zoom', 20)

        width, height = self.cam.get(3), self.cam.get(4)
        print('video size  : ', width, height)
        self.frame_idx = 0
        self.vary = None

    def run(self):
        with open('./instability_graph.csv', 'w+') as fp:
            # 进行节点稳定性信息的保存
            fp.write('frame index,node position,ent\n')
        width, height = self.cam.get(3), self.cam.get(4)
        index = -1
        while True:
            index += 1
            _ret, frame = self.cam.read()
            if _ret is not True:
                return
            if index % 6 != 0:
                pass
                # continue
            start_time = time.time()  # 耗时计算
            filter_kernel = 15
            frame = cv.blur(frame, (filter_kernel, filter_kernel))  # 对画面进行 均值滤波
            # 一些参数：渲染半径 与 速度计算半径
            render_radius = 5
            cal_radius = 5
            axis_i = list(range(cal_radius, height - cal_radius, cal_radius * 2))
            axis_j = list(range(cal_radius, width - cal_radius, cal_radius * 2))
            # 所有点位的x，y轴位置
            posis = []  # 存储所有node的position
            plots = []  # 存储所有node的即时local color
            for i in axis_i:
                for j in axis_j:
                    # 遍历两个方向轴，得到所有点位的坐标
                    posis.append((i, j))
            if not self.vary:  # 第一次进行vary的初始化
                self.vary = [[] for _ in range(len(posis))]
            # 对于所有的node 进行局部速度计算
            for posi in posis:
                c = local_color_from_render(posi, frame, cal_radius)
                c = [int(cc) if cc > 0 else 0 for cc in c]
                plots.append(c)
            # 局部速度可视化 同时更新 vary：存储过去一段时间的速度变化，用于计算信息熵
            for i in range(len(posis)):
                posi = posis[i]
                hue, sat, val = rgb_to_hsv(plots[i][0], plots[i][1], plots[i][2])
                hue = int(hue // 30)  # 速度方向分箱 的 信息熵 ：也可以计算速度大小分箱的信息熵 即val
                if len(self.vary[i]) < 20:  # 如果速度变化不足20个，则先不计算信息熵
                    self.vary[i].append(hue)
                else:
                    self.vary[i] = self.vary[i][1:] + [hue]
                    ent = calc_ent(np.array(self.vary[i]))
                # print(np.array(self.vary[i]),ent)
                    with open('./instability_graph.csv', 'a+') as fp:
                        fp.write(str(self.frame_idx) + ',' + str(posi[0]) + ';' + str(posi[1]) + ',' + str(ent) + '\n')

                cv.circle(frame, (posi[1], posi[0]), render_radius, plots[i], -1)

            sep_time = time.time() - start_time
            fr = round(1 / (sep_time), 1) if sep_time else 'inf'
            print('\rframe_rate : ', fr, end=' fps ')

            self.frame_idx += 1

            cv.namedWindow('lk_track', 0)
            cv.imshow('lk_track', frame)
            ch = cv.waitKey(3)
            if ch == ord(' '):  # quit
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = "sim_data"
        # video_src = "./sim2_of.avi"
        # video_src = "./sim3_of.avi"
        # video_src = "./sng_of.avi"
    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    # print(__doc__)
    # sample_points_from_render([50,50],0)
    main()
    cv.destroyAllWindows()
