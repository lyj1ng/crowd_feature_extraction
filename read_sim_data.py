import cv2
import numpy as np
import math


def hsv_to_rgb(h, s, v):
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


'''def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
'''


def read_sim_data():
    zoom_in = 24
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
            background = np.zeros((zoom_in*10, zoom_in*10, 3))
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
                # mag, ang = cv2.cartToPolar(vx, vy)
                mag = np.sqrt(vy ** 2 + vx ** 2)
                if round(mag, 4) == 0:
                    r, g, b = (0, 0, 0)
                else:
                    ang = np.arctan(vy / vx) * (180 / np.pi)
                    ang = (ang + 360) % 360
                    # hsv = [255] * 3
                    # hsv[0] = ang * 180 / np.pi / 2
                    # hsv[2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    # print(vx, vy)
                    mag = max(0, min(mag*2, 1))
                    # print(mag)
                    r, g, b = hsv_to_rgb(ang * 180 / np.pi / 2, 1.0, mag)
                radius = int(0.4*zoom_in)
                cv2.circle(background, (x, y), radius, (b, g, r), -1)

            cv2.imshow('frame2', background)
            k = cv2.waitKey(100) & 0xff
            if k == ord('q'):  # quit
                break


if __name__ == '__main__':
    read_sim_data()
