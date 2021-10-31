import cv2
import numpy as np
import cmath
from utils import *
import matplotlib.pyplot as plt


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


def show_sim_data(folder='sim_data', output=False):
    """
    render simulation data and measure
    :param folder: which folder to be opened as simulation data
    :return: None:output render frame and measurement
    """
    zoom_in = 15  # 视频的放大倍数
    # record size 形为 ( y , x )
    record_size = (40, 90)
    # record_size = (18, 100)
    radius = int(0.28 * zoom_in)  # 调整行人的身体半径显示大小  # previous value:0.38略挤但融合
    ax, ay = [], []
    ay2 = []
    plt.ion()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if output:
        out = cv2.VideoWriter('./sim_demo.avi', fourcc, 10,
                              (int(record_size[1] * zoom_in), int(record_size[0] * zoom_in)))
    velocity_data = []
    for frame_indx in range(400, 1210):  # 控制读几帧画面
        with open(folder + '/' + str(frame_indx) + '.xml', 'r') as fp:
            agents = []  # 存储每个agent的画图信息
            for line in fp.readlines()[3:-2]:  # 读取xml内容
                z_position = line[line.index('z') + 3:line.index('z') + 4]
                velocity = line.split()[1] + line.split()[2]
                velocity = velocity[velocity.index('{') + 1:velocity.index('}')].split(';')
                velocity = [float(i) for i in velocity]
                position = line.split()[3] + line.split()[4]
                position_x = float(position[position.index('x') + 3:position.index('x') + 7])
                y_fix_up = 9 if z_position == '0' else 7
                y_position_fix = -1 if z_position == '0' else 1
                position_y = position[position.index('y') + 3:position.index('y') + y_fix_up]
                position_y = float(position_y.replace('"', ''))
                x = int(position_x * zoom_in)  # scale position 0-10 to 0-480
                y = y_position_fix * int(position_y * zoom_in)

                agents.append([(x, y), (position_x, position_y), velocity])
                # agent的内容为【像素坐标，仿真坐标，仿真瞬时速度】

            background = np.full((zoom_in * record_size[0], zoom_in * record_size[1], 3),
                                 255, np.uint8)  # 创建黑色背景 10为q在仿真中截取的大小
            # 计算群体压力指标：先计算局部速度和局部密度
            print('(', frame_indx, end=' frame)\t')
            # test point : *Configuration*
            # position_to_test = [15, -31.5]  # 出口层的一个入口
            position_to_test = [12, -17]  # 出口层的内测
            # position_to_test = [15.5, -20]  # 出口层的一个出口
            position_z_test = 0  # 1是看台层 ，0是出口层
            test_radius = 0.7
            # calculate measurement
            y_position_fix = -1 if position_z_test == 0 else 1
            position_in_video = [int(position_to_test[0] * zoom_in),
                                 y_position_fix * int(position_to_test[1] * zoom_in)]
            cv2.circle(background, position_in_video, int(test_radius * zoom_in),
                       (0, 0, 255), -1)
            local_density, local_velocity = local_measure(position_to_test, test_radius, agents)

            if len(velocity_data) < 20:  # 此处的参数为时间间隔t用于控制计算速度方差
                velocity_data.append(local_velocity)
                velocity_variance = 0
                # last_velocity = local_velocity
            else:
                velocity_data = velocity_data[1:]+[local_velocity]
                # print(velocity_data)
                velocity_variance = cal_velocity_variance(velocity_data)

            pressure = velocity_variance * local_density
            print(round(pressure, 4), round(local_density, 2))  # , local_velocity, )
            # draw simulation
            for data in agents:
                b, g, r = 200, 200, 200  # body color
                head = (200, 100, 100)  # head color
                cv2.circle(background, data[0], radius, (b, g, r), -1)
                cv2.circle(background, data[0], int(radius * 0.5), head, -1)
            # (optional) draw optical flow
            cal_optical_flow = False
            if cal_optical_flow:
                if frame_indx == 1:
                    prev_frame = background
                    continue
                flow = cv2.calcOpticalFlowFarneback(prev_frame[:, :, 2], background[:, :, 2], None, 0.5, 3, 15, 3, 7,
                                                    1.5, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros((record_size[0] * zoom_in, record_size[1] * zoom_in, 3), np.uint8)
                hsv[..., 1] = 255
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                imgs_show = np.hstack([rgb, background])
                # out.write(rgb)
                # out2.write(imgs_show)
                # background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                if output:
                    out.write(background)
                cv2.imshow('frame2', imgs_show)
                prev_frame = background
            else:
                if output:
                    out.write(background)
                cv2.imshow('frame1', background)
            ax.append(frame_indx)
            ay.append(pressure)
            # ay2.append(local_density)
            plt.clf()
            plt.plot(ax, ay)
            plt.plot(ax, [0.02] * len(ax), linestyle='dashed')
            # plt.plot(ax, ay2)
            if frame_indx % 20 == 10:
                plt.pause(0.001)
            plt.ioff()
            # plt.show()
            k = cv2.waitKey(15) & 0xff
            if k == ord('q') or k == ord(' '):  # quit
                print('stop frame index : ', frame_indx)
                # debug information
                # for a in agents:
                #     print(a)
                break
    if output:
        out.release()
    cv2.destroyAllWindows()
    # plt.plot(ax, ay)
    # plt.plot(ax, [0.02] * len(ax), linestyle='dashed')
    plt.show()


if __name__ == '__main__':
    show_sim_data(folder='bad_situation')
    # show_sim_data()
