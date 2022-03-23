import numpy as np
import cv2, time
from utils import cosine_similarity, constrain_max_velocity

# parameter settings
lk_params = dict(winSize=(35, 35),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=30,
                      qualityLevel=0.01,
                      minDistance=7,
                      blockSize=7)

print('\ncv2 status : ', cv2.useOptimized())

cap = cv2.VideoCapture('./stopandgo.mp4')

# 获取第一帧
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
print(hsv.dtype)
# 遍历每一行的第1列
hsv[..., 1] = 255
width, height = cap.get(3), cap.get(4)
print('video size  : ', width, height)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('./track_flow.avi', fourcc, 10, (int(width), int(height)))
with open('./wave_feature_save.csv', 'w+') as fp:
    fp.write('node attribute,position,V point,V wave,video index\n')
track_len = 3
detect_interval = 5
tracks = []
stable_nodes = []
frame_idx = 0

index = -1
while True:
    ret, frame2 = cap.read()
    if not ret:
        print('\n\'\'\'End of File\'\'\'\n')
        break
    index += 1
    if index % 3 != 0:  # 每 N 帧计算一次瞬时速度
        continue
    start_time = time.time()
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 7, 1.5, 0)
    # prev 输入8位单通道图片  next 下一帧图片，格式与prev相同  flow 与返回值相同，得到一个CV_32FC2格式的光流图，与prev大小相同
    # pyr_scale 构建图像金字塔尺度 levels 图像金字塔层数  winsize 窗口尺寸，值越大探测高速运动的物体越容易，但是越模糊，同时对噪声的容错性越强
    # iterations 对每层金字塔的迭代次数  poly_n 每个像素中找到多项式展开的邻域像素的大小。越大越光滑，也越稳定
    # poly_sigma 高斯标准差，用来平滑倒数   flags 光流方式，如FARNEBACK_GAUSSIAN

    # 笛卡尔坐标转换为极坐标，获得极轴和极角
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # print(rgb[0][0])
    if index > 100:
        break
    elif index == 1:
        prev_gray = rgb
    else:
        frame = rgb
        if index % 6 != 0:
            pass
            # continue
        # start_time = time.time()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 均值滤波
        filter_kernel = 2
        frame_gray = cv2.blur(frame_gray, (filter_kernel, filter_kernel))
        vis = frame.copy()
        vis = cv2.blur(vis, (filter_kernel, filter_kernel))

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            d = abs(p1 - p0).reshape(-1, 2).max(-1)
            good = d > 1

            new_tracks = []
            show_tracks = []
            nodes = []
            graph = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
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
                    stable_nodes.append(velocity_wave)
                    nodes.append(0)

                if -0.7 < cos_wave < 0.7:
                    pass  # 不展示但是可以继续跟踪
                else:
                    nodes.append(round(cos_wave))
                    graph.append([round(cos_wave), (x, y), velocity_point, velocity_wave])
                    show_tracks.append(tr)
                    x, y = int(x), int(y)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                new_tracks.append(tr)
            tracks = new_tracks
            # print(sum(nodes), ':', len([n for n in nodes if n == -1]), len([n for n in nodes if n == 0]),
            #      len([n for n in nodes if n == 1]))

            # print(new_tracks)
            # input()
            cv2.polylines(vis, [np.int32(tr) for tr in show_tracks], False, (200, 200, 200), 2)  # show track
            with open('./wave_feature_save.csv', 'a+') as fp:
                for frame_graph in graph:
                    # fp.write('Frame ' + str(index) + '\n')
                    fp.write(str(frame_graph[0]) + ',' + str(frame_graph[1][0]) + '，' + str(frame_graph[1][1]) +
                             ',' + str(frame_graph[2][0]) + '，' + str(frame_graph[2][1]) +
                             ',' + str(frame_graph[3][0]) + '，' + str(frame_graph[3][1]) +
                             ',' + str(index) +
                             '\n')

        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])

        # sep_time = time.time() - start_time
        # print('\rframe_rate : ', round(1 / sep_time, 1), end=' fps ')

        frame_idx += 1
        prev_gray = frame_gray
        out.write(vis)
        cv2.namedWindow('lk_track', 0)
        # cv.resizeWindow('lk_track', 640, 480)
        cv2.imshow('lk_track', vis)
        # cv.imshow('lk_track', result)
        ch = cv2.waitKey(10)
        if ch == ord('q') or ch == ord(' '):  # quit
            break

    sep_time = time.time() - start_time
    print('\rframe_rate : ', round(1 / sep_time, 1), end=' fps ' + '.' * (index // 3 % 4) + ' ' * 10)


    prvs = next_frame

cap.release()

cv2.destroyAllWindows()
