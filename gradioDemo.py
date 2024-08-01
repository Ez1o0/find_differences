import gradio as gr
import cv2
import numpy as np

def find_differences(image1_path, image2_path):
    # 读取图片
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算差异并阈值化
    difference = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # 定义膨胀和腐蚀的核
    kernel = np.ones((5, 5), np.uint8)

    # 应用闭操作：先膨胀后腐蚀
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找闭操作结果中的轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # 在原图上画矩形框标记差异
    # for contour in contours:
    #     # x, y, w, h = cv2.boundingRect(contour)
    #     # cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #     # 或者直接画轮廓
    #     cv2.drawContours(image1, [contour], -1, (0, 0, 255), 2)

    # 在原始图像上标记轮廓的最小外接矩形
    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 获取最小外接矩形
        box = cv2.boxPoints(rect)  # 计算矩形的四个顶点
        box = np.int32(box)  # 转换为整数
        cv2.drawContours(image1, [box], 0, (0, 0, 255), 2)  # 用红色标记矩形


    # 将处理后的图像转换为RGB格式，因为OpenCV使用BGR格式
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    return image1

# 创建Gradio接口
demo = gr.Interface(
    fn=find_differences,
    inputs=[gr.Image(label="上传图片1", type="filepath"), gr.Image(label="上传图片2", type="filepath")],
    outputs=gr.Image(label="处理后的图片"),
    title="查找两种图片的不同",
    description="上传两张图片",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()