import cv2
import numpy as np
import cmath
from utils import *
import matplotlib.pyplot as plt
from random import random


def hsv_to_rgb(h, s, v):
    '''
    :param h: hue 色相
    :param s: 饱和度
    :param v: magnitude
    :return: instance like
    hsv_to_rgb(359,1,1)
    [1, 0.0, 0.0]
    '''
    if s == 0.0: return v, v, v
    i = int(h * 6.)  # XXX assume int() truncates!
    f = (h * 6.) - i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    if i == 5: return v, p, q


def local_measure(position, radius, agents):
    """
    calculate local density and velocity
    :param position: position to measure:[0,0](position in modeling)
    :param radius: radius R of measure:the bigger R,the smoothing area around position
    :param agents: data of all agents:agent[0] is position and agent[1] is velocity
    :return: local density and local velocity
    """
    local_density = 0
    local_velocity = np.zeros(2)
    for agent in agents:
        weight = np.exp(-1 * euclid_distance(agent[1], position, root=False) / (radius ** 2))
        local_density += weight
        local_velocity += weight * np.array(agent[2])
    local_velocity /= local_density
    local_density = local_density / (3.1416 * radius * radius)
    return local_density, local_velocity


def show_sim_data(folder='sim_data', output=False, show=False):
    """
    render simulation data and measure
    :param folder: which folder to be opened as simulation data
    :return: None:output render frame and measurement
    """
    zoom_in = 20  # 视频的放大倍数
    # record size 形为 ( y , x )
    record_size = (40, 70)
    # record_size = (18, 100)
    radius = int(0.28 * zoom_in)  # 调整行人的身体半径显示大小  # previous value:0.38略挤但融合
    ax, ay = [], []
    ay2 = []
    plt.ion()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if output:
        out = cv2.VideoWriter('./sim_demo.avi', fourcc, 10,
                              (int(record_size[1] * zoom_in), int(record_size[0] * zoom_in)))
    save_name = 'label_' + folder.split('/')[-1] + '_' + str(int(random() * 1000)) + '.csv'
    with open(save_name, 'w+') as sfp:
        sfp.write('frame index,label,node position:')
    write_position = 1
    velocity_data = []
    for frame_indx in range(10, 3200):  # 控制读几帧画面
        with open(folder + '/' + str(frame_indx) + '.xml', 'r') as fp:
            agents = []  # 存储每个agent的画图信息
            for line in fp.readlines()[3:-2]:  # 读取xml内容
                z_position = line[line.index('z') + 3:line.index('z') + 4]
                velocity = line.split()[1] + line.split()[2]
                velocity = velocity[velocity.index('{') + 1:velocity.index('}')].split(';')
                velocity = [float(i) for i in velocity]
                position = line.split()[3] + line.split()[4]
                position_x = position[position.index('x') + 3:position.index('x') + 8]
                position_x = float(position_x.replace('"', ''))
                y_fix_up = 9 if z_position == '0' else 7
                y_position_fix = -1 if z_position == '0' else 1
                position_y = position[position.index('y') + 3:position.index('y') + y_fix_up]
                position_y = float(position_y.replace('"', ''))
                x = int(position_x * zoom_in)  # scale position 0-10 to 0-480
                y = y_position_fix * int(position_y * zoom_in)
                # print(position_y,position_x)
                agents.append([(x, y), (position_x, position_y), velocity])
                # agent的内容为【像素坐标，仿真坐标，仿真瞬时速度】

            background = np.full((zoom_in * record_size[0], zoom_in * record_size[1], 3),
                                 255, np.uint8)  # 创建黑色背景 10为q在仿真中截取的大小
            # draw simulation
            for data in agents:
                b, g, r = 200, 200, 200  # body color
                head = (200, 100, 100)  # head color
                cv2.circle(background, data[0], radius, (b, g, r), -1)
                cv2.circle(background, data[0], int(radius * 0.5), head, -1)
            # 计算群体压力指标：先计算局部速度和局部密度

            print('(', frame_indx, end=' frame)\t')
            # test point : *Configuration*
            # position_to_test = [15, -31.5]  # 出口层的一个入口
            position_to_test = [12, -17]  # 出口层的内侧
            # position_to_test = [11, -25]  # crowd but laminar first and laminar last
            # position_to_test = [13.5, -31.1]  # 交替
            # position_to_test = [8.5, -25.8]  # 3300frame左右
            # position_to_test = [12, -17]
            # position_to_test = [17.5, -20]  # 出口层的一个出口
            # position_to_test = [20, -30.5]  # laminar position
            position_z_test = 0  # 1是看台层 ，0是出口层

            test_radius = 0.7

            # calculate measurement
            y_position_fix = -1 if position_z_test == 0 else 1
            position_in_video = [int(position_to_test[0] * zoom_in),
                                 y_position_fix * int(position_to_test[1] * zoom_in)]
            if frame_indx % 10 < 5:  # 视觉效果
                cv2.circle(background, position_in_video, int(test_radius * zoom_in),
                           (50, 50, 253), 3)
            local_density, local_velocity = local_measure(position_to_test, test_radius, agents)

            if frame_indx % 1 == 0:  # 控制速度的采样间隔
                if len(velocity_data) < 25:  # 此处的参数为时间间隔t用于控制计算速度方差
                    # t越小，pressure变化曲线越不平滑，得到的pressure越小
                    velocity_data.append(local_velocity)
                    velocity_variance = 0
                else:
                    velocity_data = velocity_data[1:] + [local_velocity]
                    # print(velocity_data)
                    velocity_variance = cal_velocity_variance(velocity_data)

            pressure = velocity_variance * local_density
            print(round(pressure, 4), round(local_density, 2))  # , local_velocity, )
            label = 0
            if pressure > 0.08 and local_density > 5:
                label = 1
            if write_position:
                with open(save_name, 'a') as sfp:
                    sfp.write(str(position_to_test) + '\n')
                write_position = 0
            with open(save_name, 'a') as sfp:
                sfp.write(str(frame_indx) + ',' + str(label) + '\n')

            if output:
                out.write(background)
            if show:
                cv2.imshow('frame1', background)
                if pressure:
                    ax.append(frame_indx / 10)
                    ay.append(pressure)
                # ay2.append(local_density)
                plt.clf()
                plt.title('CROWD PRESSURE of the circled area\n'
                          + 'current density : ' + str(round(local_density, 2))
                          + '\ncurrent pressure : ' + str(round(pressure, 4))
                          )
                l1, = plt.plot(ax, ay)
                l2, = plt.plot(ax, [0.08] * len(ax), linestyle='dashed')
                plt.legend([l1, l2], ['crowd pressure', 'threshold of 0.08'], loc='upper right')
            if True or frame_indx % 10 == 5:
                plt.pause(0.001)
            plt.ioff()
            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == ord(' '):  # quit
                print('stop frame index : ', frame_indx)
                # debug information
                # for a in agents:
                #     print(a)
                break
    if output:
        out.release()
    cv2.destroyAllWindows()
    plt.show()


if __name__ == '__main__':
    # show_sim_data(folder='bad_situation', output=False)
    # show_sim_data(folder='D:/simulation/congestion', output=False)
    show_sim_data(folder='D:/simulation/turb', output=False)
    # show_sim_data()
