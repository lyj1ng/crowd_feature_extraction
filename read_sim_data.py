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


def read_sim_data(folder='sim_data'):
    zoom_in = 15  # 视频的放大倍数
    # record size 形为 ( y , x )
    # record_size = (10, 10) 看台(18,100)
    record_size = (50, 100)
    radius = int(0.28 * zoom_in)  # 调整行人的身体半径显示大小  # previous value:0.38略挤但融合

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./sim_to_optical_flow.avi', fourcc, 10,
                          (int(record_size[1] * zoom_in), int(record_size[0] * zoom_in)))

    for frame_indx in range(1, 900):  # 控制读几帧画面
        with open(folder + '/' + str(frame_indx) + '.xml', 'r') as fp:
            agents = []  # 存储每个agent的画图信息
            mags = []  # 用于后续对速度进行正则化显示
            for line in fp.readlines()[3:-2]:  # 读取xml内容

                z_position = line[line.index('z') + 3:line.index('z') + 4]
                # print()
                velocity = line.split()[1] + line.split()[2]
                velocity = velocity[velocity.index('{') + 1:velocity.index('}')].split(';')
                velocity = [float(i) for i in velocity]
                position = line.split()[3] + line.split()[4]
                # print(position)
                position_x = position[position.index('x') + 3:position.index('x') + 7]
                y_fix_up = 9 if z_position=='0' else 7
                y_position_fix = -1 if z_position=='0' else 1
                position_y = position[position.index('y') + 3:position.index('y') + y_fix_up]
                position_y = position_y.replace('"','')
                # print(position_y,position_x)
                # 接下来对agent数据的操作包括
                # position到像素的整理
                # 速度到HSV图像的转换
                x = int(float(position_x) * zoom_in)  # scale position 0-10 to 0-480
                y = y_position_fix*int(float(position_y) * zoom_in)
                # scale velocity
                # turn velocity into color
                vx = velocity[0]
                vy = y_position_fix*-1*velocity[1]
                cn = complex(vx, vy)
                mag, ang = cmath.polar(cn)
                mags.append(mag)
                # agents.append([(x, y), ang, mag])
                # ang /= 3.14*2
                # ang += 1
                ang = (ang / (3.1416 * 2) + 1) % 1
                agents.append([(x, y), ang, mag, velocity, position, hsv_to_rgb(ang, 1.0, mag)])

            background = np.zeros((zoom_in * record_size[0], zoom_in * record_size[1], 3),
                                  np.uint8)  # 创建黑色背景 10为q在仿真中截取的大小
            max_mag, min_mag = np.max(mags), np.min(mags)
            # mu, sigma = np.mean(mags),np.std(mags)
            for data in agents:
                posi = data[0]
                ang = data[1]
                mag = data[2]
                if max_mag == min_mag:
                    mag = 0
                else:
                    mag = (mag - min_mag) / (max_mag - min_mag)
                mag *= 3
                # mag = (mag - mu) / sigma
                r, g, b = hsv_to_rgb(ang, 1.0, mag)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                # 记录一下这里：如果图片格式是float，那么范围在0-1；如果图片格式为int，范围在0-255。
                # 为了后面保存 or 其他操作的需要（因为opencv无法操作float64），这里将float转为int因此要乘上255
                cv2.circle(background, posi, radius, (b, g, r), -1)

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
            k = cv2.waitKey(30) & 0xff
            if k == ord('q') or k == ord(' '):  # quit
                print(frame_indx)
                # debug information
                # for a in agents:
                #     print(a)
                break
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # read_sim_data(folder='full_size')
    read_sim_data(folder='bad_situation')
    # read_sim_data()
