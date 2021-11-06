#!/usr/bin/env python

import numpy as np
import cv2 as cv
from utils import *
import time
from sim_data_iterator import SimData


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
            fp.write('frame index,node position,ent,mutual info:up,mutual info:down,mutual info:left,mutual info:right\n')
        width, height = self.cam.get(3), self.cam.get(4)
        index = -1

        render_radius = 5
        cal_radius = 5
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

            plots = []  # 存储所有node的即时local color
            if not self.vary:  # 第一次进行vary的初始化
                self.vary = [[] for _ in range(len(posis))]
            # 对于所有的node 进行局部速度计算
            for posi in posis:
                c = local_color_from_render((posi[1], posi[0]), frame, cal_radius)
                c = [int(cc) if cc > 0 else 0 for cc in c]
                plots.append(c)
            # 局部速度可视化 同时更新 vary：存储过去一段时间的速度变化，用于计算信息熵
            for i in range(len(posis)):
                posi = posis[i]
                hue, sat, val = rgb_to_hsv(plots[i][1], plots[i][2], plots[i][0])
                hue = int(hue // 30)  # 速度方向分箱 的 信息熵 ：也可以计算速度大小分箱的信息熵 即val
                if len(self.vary[i]) < 20:  # 如果速度变化不足20个，则先不计算信息熵
                    self.vary[i].append(hue)
                else:
                    # 数组的拼接
                    self.vary[i] = self.vary[i][1:] + [hue]
                    # 计算信息熵
                    ent = calc_ent(np.array(self.vary[i]))
                    # 计算互信息：上下左右
                    mutual_info = []
                    for neighbor in [[0, -2 * cal_radius], [0, 2 * cal_radius], [-2 * cal_radius, 0],
                                     [2 * cal_radius, 0]]:
                        neighbor_position = np.array(posi) + np.array(neighbor)
                        neighbor_position = [str(p) for p in neighbor_position]
                        neighbor_position = ';'.join(neighbor_position)
                        # print(neighbor_position,position_map)
                        if neighbor_position in position_map:
                            n_vary = np.array(self.vary[position_map[neighbor_position]])
                            mutual_info.append(calc_ent_grap(np.array(self.vary[i]), n_vary))
                        else:
                            mutual_info.append(None)

                    with open('./instability_graph.csv', 'a+') as fp:
                        fp.write(str(self.frame_idx) + ',' + str(posi[0]) + ';' + str(posi[1])
                                 + ',' + str(ent) + ',' + str(mutual_info) + '\n')

                cv.circle(frame, (posi[0], posi[1]), render_radius, plots[i], -1)

            # real-time frame fresh rate
            sep_time = time.time() - start_time
            fr = round(1 / sep_time, 1) if sep_time else 'inf'
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
