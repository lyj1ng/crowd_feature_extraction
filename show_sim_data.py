import cv2
import numpy as np
import cmath
from utils import *


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


def cal_partial_velocity(something):
    return


def show_sim_data(folder='sim_data'):
    zoom_in = 30  # 视频的放大倍数
    # record size 形为 ( y , x )
    record_size = (10, 10)
    # record_size = (18, 100)
    radius = int(0.28 * zoom_in)  # 调整行人的身体半径显示大小  # previous value:0.38略挤但融合

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./sim_demo.avi', fourcc, 10,
                          (int(record_size[1] * zoom_in), int(record_size[0] * zoom_in)))

    for frame_indx in range(1, 900):  # 控制读几帧画面
        with open(folder + '/' + str(frame_indx) + '.xml', 'r') as fp:
            agents = []  # 存储每个agent的画图信息
            # mags = []  # 用于后续对速度进行正则化显示
            for line in fp.readlines()[3:-2]:  # 读取xml内容
                velocity = line.split()[1] + line.split()[2]
                velocity = velocity[velocity.index('{') + 1:velocity.index('}')].split(';')
                velocity = [float(i) for i in velocity]
                position = line.split()[3] + line.split()[4]
                position_x = float(position[position.index('x') + 3:position.index('x') + 7])
                position_y = float(position[position.index('y') + 3:position.index('y') + 7])

                x = int(position_x * zoom_in)  # scale position 0-10 to 0-480
                y = int(position_y * zoom_in)

                agents.append([(x, y), (position_x, position_y), velocity])
                # agent的内容为【像素坐标，仿真坐标，仿真瞬时速度】

            background = np.full((zoom_in * record_size[0], zoom_in * record_size[1], 3),
                                 255, np.uint8)  # 创建黑色背景 10为q在仿真中截取的大小
            # 计算群体压力指标：先计算局部速度和局部密度
            print(frame_indx, end=' ')
            position_to_test = [5, 5]
            test_radius = 1
            cv2.circle(background, [int(i*zoom_in) for i in position_to_test], int(test_radius*zoom_in), (0, 0, 0), -1)
            velo_sum = np.zeros(2)
            times = 0
            for agent in agents:
                posi = agent[1]
                if euclid_distance(posi,position_to_test) < test_radius:
                    velo_sum += agent[2]
                    times += 1
            if times:
                velo_sum /= times
            print(velo_sum,times)


            # max_mag, min_mag = np.max(mags), np.min(mags)
            # mu, sigma = np.mean(mags),np.std(mags)
            for data in agents:
                posi = data[0]

                b, g, r = 200, 200, 200  # body color
                head = (200, 100, 100)  # head color
                cv2.circle(background, posi, radius, (b, g, r), -1)
                cv2.circle(background, posi, int(radius * 0.5), head, -1)

            # 至此
            # 每帧 仿真 对应 的 运动场 渲染 完毕 ：cv2.imshow('frame xxx', background)可以进行输出
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
                out.write(background)
                cv2.imshow('frame2', imgs_show)
                prev_frame = background
            else:
                out.write(background)
                cv2.imshow('frame1', background)
            k = cv2.waitKey(10) & 0xff
            if k == ord('q') or k == ord(' '):  # quit
                print('stop frame index : ', frame_indx)
                # debug information
                # for a in agents:
                #     print(a)
                break
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # show_sim_data(folder='full_size')
    show_sim_data()
