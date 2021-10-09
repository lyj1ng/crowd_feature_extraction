import cv2
import numpy as np
import cmath


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


def read_sim_data():
    zoom_in = 20  # 视频的放大倍数
    radius = int(0.38 * zoom_in)  # 调整行人的身体半径显示大小
    for frame_indx in range(1, 900):  # 控制读几帧画面
        with open('sim_data/' + str(frame_indx) + '.xml', 'r') as fp:
            agents = []  # 存储每个agent的画图信息
            mags = []  # 用于后续对速度进行正则化显示
            for line in fp.readlines()[3:-2]:  # 读取xml内容
                velocity = line.split()[1] + line.split()[2]
                velocity = velocity[velocity.index('{') + 1:velocity.index('}')].split(';')
                velocity = [float(i) for i in velocity]
                position = line.split()[3] + line.split()[4]
                position_x = position[position.index('x') + 3:position.index('x') + 7]
                position_y = position[position.index('y') + 3:position.index('y') + 7]
                # 接下来对agent数据的操作包括
                # position到像素的整理
                # 速度到HSV图像的转换
                x = int(float(position_x) * zoom_in)  # scale position 0-10 to 0-480
                y = int(float(position_y) * zoom_in)
                # scale velocity
                # turn velocity into color
                vx = velocity[0]
                vy = velocity[1]
                cn = complex(vx, vy)
                mag, ang = cmath.polar(cn)
                mags.append(mag)
                agents.append([(x, y), ang, mag])

            background = np.zeros((zoom_in * 10, zoom_in * 10, 3))  # 创建黑色背景 10为在仿真中截取的大小
            max_mag, min_mag = np.max(mags), np.min(mags)
            # mu, sigma = np.mean(mags),np.std(mags)
            for data in agents:
                posi, ang, mag = data
                mag = (mag - min_mag) / (max_mag - min_mag)
                # mag = (mag - mu) / sigma
                r, g, b = hsv_to_rgb(ang, 1.0, mag)
                cv2.circle(background, posi, radius, (b, g, r), -1)

            cv2.imshow('frame2', background)
            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == ord(' '):  # quit
                break


if __name__ == '__main__':
    read_sim_data()
