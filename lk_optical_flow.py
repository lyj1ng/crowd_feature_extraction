#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2 as cv
from utils import cosine_similarity, constrain_max_velocity
import time

# from time import clock

lk_params = dict(winSize=(35, 35),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=30,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):
        self.track_len = 3
        self.detect_interval = 5
        self.tracks = []
        self.stable_nodes = []
        self.cam = cv.VideoCapture(video_src)
        width, height = self.cam.get(3), self.cam.get(4)
        print('video size  : ', width, height)
        self.frame_idx = 0

    def run(self):
        with open('./graph.csv', 'w+') as fp:
            fp.write('node attribute,position,V point,V wave,video index\n')
        width, height = self.cam.get(3), self.cam.get(4)
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter('./flow.avi', fourcc, 10, (int(width), int(height)))
        index = -1
        while True:
            index += 1
            _ret, frame = self.cam.read()
            if _ret is not True:
                return
            if index % 6 != 0:
                pass
                # continue

            start_time = time.time()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  #################
            # print(frame_gray.shape)
            filter_kernel = 2
            frame_gray = cv.blur(frame_gray, (filter_kernel, filter_kernel))  # 均值滤波
            vis = frame.copy()
            vis = cv.blur(vis, (filter_kernel, filter_kernel))  # 均值滤波
            # print(vis)
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # print(p1)
                d = abs(p1 - p0).reshape(-1, 2).max(-1)
                good = d > 1

                new_tracks = []
                show_tracks = []
                nodes = []
                graph = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    try:
                        bgr_vector = vis[np.int32(y)][np.int32(x)]  # 获得了追踪点所代表的颜色 ,BGR通道
                    except:
                        continue  # 对于超出图像范围的追踪点 进行舍弃
                    if not constrain_max_velocity(np.array(tr[-1]) - np.array(tr[-2]), threshold=800):
                        # 相当于是对 V wave 进行了限制
                        continue
                    # print(np.array(tr[-1]) - np.array(tr[-2]))

                    red_vector, green_vector, blue_vector = (1, 0), (-0.5, 0.866), (-0.5, -0.866)
                    velocity_point = bgr_vector[0] * np.array(blue_vector) + bgr_vector[1] * np.array(green_vector) + \
                                     bgr_vector[2] * np.array(red_vector)
                    velocity_wave = np.array(tr[-1]) - np.array(tr[0])
                    # print(velocity_wave, velocity_point)
                    cos_wave = cosine_similarity(velocity_wave, velocity_point)
                    if constrain_max_velocity(velocity_wave, threshold=10):  # 首先筛选稳定节点
                        self.stable_nodes.append(velocity_wave)
                        nodes.append(0)

                    if -0.7 < cos_wave < 0.7:
                        pass  # 不展示但是可以继续跟踪
                    else:
                        nodes.append(round(cos_wave))
                        graph.append([round(cos_wave), (x, y), velocity_point, velocity_wave])
                        show_tracks.append(tr)
                        x, y = int(x), int(y)
                        cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    new_tracks.append(tr)
                self.tracks = new_tracks
                # print(sum(nodes), ':', len([n for n in nodes if n == -1]), len([n for n in nodes if n == 0]),
                #      len([n for n in nodes if n == 1]))

                # print(new_tracks)
                # input()
                cv.polylines(vis, [np.int32(tr) for tr in show_tracks], False, (200, 200, 200), 2)  # show track
                with open('./graph.csv', 'a+') as fp:
                    # index = 0
                    for frame_graph in graph:
                        # fp.write('Frame ' + str(index) + '\n')
                        fp.write(str(frame_graph[0]) + ',' + str(frame_graph[1][0]) + '，' + str(frame_graph[1][1]) +
                                 ',' + str(frame_graph[2][0]) + '，' + str(frame_graph[2][1]) +
                                 ',' + str(frame_graph[3][0]) + '，' + str(frame_graph[3][1]) +
                                 ',' + str(index) +
                                 '\n')
                        # index += 1

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv.circle(mask, (x, y), 5, 0, -1)
                p = cv.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            sep_time = time.time() - start_time
            print('\rframe_rate : ', round(1 / sep_time, 1), end=' fps ')

            self.frame_idx += 1
            self.prev_gray = frame_gray
            out.write(vis)
            cv.namedWindow('lk_track', 0)
            # cv.resizeWindow('lk_track', 640, 480)
            cv.imshow('lk_track', vis)
            # cv.imshow('lk_track', result)
            ch = cv.waitKey(100)
            if ch == ord('q'):  # quit
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = "./tb_of.avi"
        # video_src = "./sim2_of.avi"
        # video_src = "./sim3_of.avi"
        # video_src = "./sng_of.avi"
    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    # print(__doc__)
    main()
    cv.destroyAllWindows()
