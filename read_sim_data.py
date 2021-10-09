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
    zoom_in = 20
    for frame_indx in range(1, 900):  # 控制读几帧画面
        with open('sim_data/' + str(frame_indx) + '.xml', 'r') as fp:
            agents = []
            for line in fp.readlines()[3:-2]:
                velocity = line.split()[1] + line.split()[2]
                velocity = velocity[velocity.index('{') + 1:velocity.index('}')].split(';')
                velocity = [float(i) for i in velocity]
                position = line.split()[3] + line.split()[4]
                position_x = position[position.index('x') + 3:position.index('x') + 7]
                position_y = position[position.index('y') + 3:position.index('y') + 7]
                position = [float(position_x), float(position_y)]
                agents.append(position + velocity)
                # break
            # print(agents)
            # 接下来对agent数据的操作包括
            # position到像素的整理
            # 速度到HSV图像的转换
            # agents = [[6,5,1,0],[4,5,-1,0],[5,6,0,1],[5,4,0,-1],[4,6,-1,-1],[4,4,-1,1]]  # 用于调试色彩空间
            background = np.zeros((zoom_in * 10, zoom_in * 10, 3))
            # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # x = np.random.randint(0,480)
            # y = np.random.randint(0,480)
            for agent in agents:
                # scale position 0-10 to 0-480
                x = int(agent[0] * zoom_in)
                y = int(agent[1] * zoom_in)
                # scale velocity
                # turn velocity into color
                vx = agent[2]
                vy = agent[3]
                # cn = complex(vx, vy)
                # print(cmath.polar(cn))
                # mag, ang = cv2.cartToPolar(vx, vy)
                mag = np.sqrt(vy ** 2 + vx ** 2)
                if round(mag, 4) == 0:
                    r, g, b = (0, 0, 0)
                else:
                    cn = complex(vx, vy)
                    mag, ang = cmath.polar(cn)
                    # ang = ang * (180 / np.pi)
                    # ang = (ang + 360) % 360
                    # if ang < 0:
                    #     print(ang)
                    #     ang += 360
                    # hsv = [255] * 3
                    # hsv[0] = ang * 180 / np.pi / 2
                    # hsv[2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    # print(vx, vy)
                    mag = max(0, min(mag * 2, 1))
                    # print(mag)
                    r, g, b = hsv_to_rgb(ang, 1.0, mag)
                    print(agent, mag, ang, r, g, b)
                radius = int(0.4 * zoom_in)

                cv2.circle(background, (x, y), radius, (b, g, r), -1)

            cv2.imshow('frame2', background)
            k = cv2.waitKey(10) & 0xff
            if k == ord('q') or k == ord(' '):  # quit
                break


if __name__ == '__main__':
    read_sim_data()
