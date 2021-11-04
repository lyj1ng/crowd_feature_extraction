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
        weight = np.exp(-1 * euclid_distance(agent[1], position, root=False) / (radius ** 2))
        local_density += weight
        local_color += weight * np.array(agent[1])
    local_color /= local_density
    return local_color


def sample_points_from_render(position, vis, sample_times=10, radius=20):
    ret = []
    vis=np.array(vis)
    for i in range(sample_times):
        xi, yi = np.random.randint(-1*radius,radius), np.random.randint(-1*radius,radius)
        sample_position = (position[0]+xi,position[1]+yi)
        # if sample_position[0]>0  # 此处待加入限制
        # print(sample_position)
        ret.append([sample_position,vis[sample_position]])
    return ret


def local_color_from_render(position,vis,radius=10):
    # position = (position[1],position[0])
    agents = sample_points_from_render(position,vis,radius=2*radius)
    return cal_local_velocity(position,agents,radius=radius)



class App:
    def __init__(self, video_src):
        self.track_len = 3
        self.detect_interval = 5
        self.tracks = []
        self.stable_nodes = []
        self.cam = SimData(video_src)
        width, height = self.cam.get(3), self.cam.get(4)
        print('video size  : ', width, height)
        self.frame_idx = 0

    def run(self):
        with open('./graph.csv', 'w+') as fp:
            fp.write('node attribute,position,V point,V wave,video index\n')
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
            # if check_sim_optical_data_only:
            #    cv.imshow('lk_track', frame)
            #    ch = cv.waitKey(100)
            #    if ch == ord('q'):  # quit
            #        break
            #    continue
            start_time = time.time()
            # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  #################
            # print(frame_gray.shape)
            filter_kernel = 15
            frame = cv.blur(frame, (filter_kernel, filter_kernel))  # 均值滤波

            kd = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
            posis = []
            for i in kd:
                for j in kd:
                    posis.append((i,j))

            #
            for posi in posis:
                c=local_color_from_render(posi, frame, 10)
                c= [int(cc) if cc>0 else 0 for cc in c]
                # print(c)

                # frame_gray = cv.blur(frame_gray, (filter_kernel, filter_kernel))  # 均值滤波
                # vis = frame.copy()

                # print(vis)
                b,g,r = c
                cv.circle(frame, posi, 5, (b,g,r), -1)



            sep_time = time.time() - start_time
            fr = round(1 / (sep_time), 1) if sep_time else 'inf'
            print('\rframe_rate : ', fr, end=' fps ')

            self.frame_idx += 1
            # self.prev_gray = frame_gray

            # cv.circle(frame, [50,50], 2,(1,1,1), -1)


            cv.namedWindow('lk_track', 0)
            cv.imshow('lk_track', frame)
            ch = cv.waitKey(10)
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
