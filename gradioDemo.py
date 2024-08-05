import gradio as gr
import cv2
import numpy as np
# from PIL import Image
# import warnings

def find_differences(image1, image2):
    # Calculate the number of pixels
    num_pixels1 = image1.shape[0] * image1.shape[1]
    num_pixels2 = image2.shape[0] * image2.shape[1]

    # Check if the number of pixels exceeds your safe threshold
    if num_pixels1 > 89478485 or num_pixels2 > 89478485:
        gr.Error("图片过大，请压缩后再上传")
        # gr.Warning("图片过大，将进行压缩")

        # # 等比例压缩图片
        # scale_factor1 = (89478485 / num_pixels1) ** 0.5
        # width1 = int(image1.shape[1] * scale_factor1)
        # height1 = int(image1.shape[0] * scale_factor1)
        # # 使用 INTER_AREA 方法进行高质量压缩
        # image1 = cv2.resize(image1, (width1, height1), interpolation=cv2.INTER_AREA)

        # # 等比例压缩图片
        # scale_factor2 = (89478485 / num_pixels2) ** 0.5
        # width2 = int(image2.shape[1] * scale_factor2)
        # height2 = int(image2.shape[0] * scale_factor2)
        # # 使用 INTER_AREA 方法进行高质量压缩
        # image2 = cv2.resize(image2, (width2, height2), interpolation=cv2.INTER_AREA)

        # if image1.shape[0] * image1.shape[1] > 89478485 or image2.shape[0] * image2.shape[1] > 89478485:
        #     raise gr.Error("图片过大，压缩后仍然超过限制")

    # 转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

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
        cv2.drawContours(image1, [box], 0, (255, 0, 0), 2)  # 用红色标记矩形

    return image1

# 创建Gradio接口
app = gr.Interface(
    fn=find_differences,
    inputs=[gr.Image(label="上传图片1", image_mode='RGB', type='numpy'), gr.Image(label="上传图片2", image_mode='RGB', type='numpy')],
    outputs=gr.Image(label="处理后的图片"),
    title="查找两种图片的不同",
    description="上传两张图片",
    allow_flagging="never"
)

if __name__ == "__main__":
    app.launch(root_path="/hqb/find_differences", debug=False)