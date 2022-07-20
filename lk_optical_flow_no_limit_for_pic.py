import cv2
import os


def image2video():
    # 得到图像路径
    files = os.listdir("C:\\Users\\forev\\Desktop\\run\\3niurunning2\\")
    # 对图像排序
    files.sort(key=lambda x: int(x.split(".")[0]))
    # 获取图像宽高
    h, w, _ = cv2.imread("C:\\Users\\forev\\Desktop\\run\\3niurunning2\\" + files[0]).shape
    # 设置帧数
    fps = 30
    vid = []
    # 保存视频路径和名称
    save_path = "pic.mp4"  # 保存视频路径和名称 MP4格式

    # 准备写入视频
    vid = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 写入
    for file in files:
        img = cv2.imread("C:\\Users\\forev\\Desktop\\run\\3niurunning2\\" + file)
        vid.write(img)


image2video()
