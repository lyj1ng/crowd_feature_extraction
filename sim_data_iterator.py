import cv2
import numpy as np
import cmath
from utils import hsv_to_rgb


class SimData:
    """
    SimData:
    the iterator of simulation data
    :param sim_src : the name of simulation data folder
    :method get    : get the pixel size of output image
    :method read   : read sim data and output sim_optical_flow like a cam (like opencv library)
    """

    def __init__(self, sim_src):
        self.folder = sim_src
        self.start_idx, self.end_idx = 10, 3000
        self.zoom_in = 25
        self.record_size = (10, 10)
        self.human_radius = 0.28
        self.idx = self.start_idx

    def set(self, config, value):
        """
        设置仿真参数
        :param config: in ['idx','zoom','size','human_radius']
        :param value: set the value of config
        :return:
        """
        if config == 'idx':
            self.start_idx = value[0]
            self.end_idx = value[1]
        elif config == 'zoom':
            self.zoom_in = value
        elif config == 'size':
            self.record_size = value
        elif config == 'human_radius':
            self.human_radius = value

    def get(self, num):
        if num == 4:
            return self.record_size[0] * self.zoom_in
        elif num == 3:
            return self.record_size[1] * self.zoom_in

    def read(self):
        folder = self.folder
        zoom_in = self.zoom_in  # 视频的放大倍数
        # record size 形为 ( y , x )
        # record_size = (10, 10) 看台(18,100)
        record_size = self.record_size
        radius = int(self.human_radius * self.zoom_in)  # 调整行人的身体半径显示大小  # previous value:0.38略挤但融合
        frame_indx = self.idx
        if self.idx > self.end_idx:
            return False, False
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
                y_fix_up = 9 if z_position == '0' else 7
                y_position_fix = -1 if z_position == '0' else 1
                position_y = position[position.index('y') + 3:position.index('y') + y_fix_up]
                position_y = position_y.replace('"', '')
                # print(position_y,position_x)
                # 接下来对agent数据的操作包括
                # position到像素的整理
                # 速度到HSV图像的转换
                x = int(float(position_x) * zoom_in)  # scale position 0-10 to 0-480
                y = y_position_fix * int(float(position_y) * zoom_in)
                # scale velocity
                # turn velocity into color
                vx = velocity[0]
                vy = y_position_fix * -1 * velocity[1]
                cn = complex(vx, vy)
                mag, ang = cmath.polar(cn)
                mags.append(mag)
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
        self.idx += 1
        return True, background


if __name__ == '__main__':
    a = SimData('sim_data')
    ret, b = a.read()
    print(ret, a.idx)
    while ret:
        ret, b = a.read()
        print(ret, a.idx)
