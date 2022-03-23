import numpy as np
import cv2, time

print('\ncv2 status : ', cv2.useOptimized())
# cap = cv2.VideoCapture('./2.mp4')
# cap = cv2.VideoCapture('./sim2.mov')
# cap = cv2.VideoCapture('./sim1.mov')
# cap = cv2.VideoCapture('./tb_of.avi')
cap = cv2.VideoCapture('./stopandgo.mp4')
# cap = cv2.VideoCapture('./turbulence.mp4')

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

out = cv2.VideoWriter('./output.avi', fourcc, 10, (int(width), int(height)))
out2 = cv2.VideoWriter('./demo.avi', fourcc, 10, (int(width) * 2, int(height)))

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
    # print(next_frame)
    # input()
    # cv2.imshow('frame2',prvs)
    # k = cv2.waitKey(300)
    # cv2.imshow('frame2',next_frame)
    # k = cv2.waitKey(3)

    # 返回一个两通道的光流向量，实际上是每个点的像素位移值
    # flow = cv2.calcOpticalFlowFarneback(prvs,next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 7, 1.5, 0)
    #                                   prev 输入8位单通道图片
    #                                        next 下一帧图片，格式与prev相同
    #                                                   flow 与返回值相同，得到一个CV_32FC2格式的光流图，与prev大小相同
    #                                                          pyr_scale 构建图像金字塔尺度
    #                                                              levels 图像金字塔层数
    #                                                                  winsize 窗口尺寸，值越大探测高速运动的物体越容易，但是越模糊，同时对噪声的容错性越强
    #                                                                      iterations 对每层金字塔的迭代次数
    #                                                                          poly_n 每个像素中找到多项式展开的邻域像素的大小。越大越光滑，也越稳定
    #                                                                              poly_sigma 高斯标准差，用来平滑倒数
    #                                                                                 flags 光流方式，如FARNEBACK_GAUSSIAN

    # print(flow.shape)
    # print(flow)

    # 笛卡尔坐标转换为极坐标，获得极轴和极角
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show_frame = cv2.cvtColor(next_frame, cv2.COLOR_GRAY2BGR)
    imgs_show = np.hstack([rgb, show_frame])

    out.write(rgb)
    out2.write(imgs_show)

    sep_time = time.time() - start_time
    print('\rframe_rate : ', round(1 / sep_time, 1), end=' fps ' + '.' * (index // 3 % 4) + ' ' * 10)

    cv2.namedWindow('frame2', 0);
    cv2.resizeWindow('frame2', 1280, 480);
    cv2.imshow('frame2', imgs_show)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):  # quit
        break
    elif k == ord('s'):  # save
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = next_frame

cap.release()
out.release()
cv2.destroyAllWindows()
