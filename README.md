# EYE BLINK DETECTION WITH OPENCV AND MEDIAPIPE

### 仅使用 Python、OpenCV 和 mediapipe 检测一个人是否在眨眼

参考资料:

[官方文档](https://chuoling.github.io/mediapipe/solutions/face_mesh.html)

[EYE BLINK DETECTION WITH OPENCV AND DLIB](https://github.com/Practical-CV/EYE-BLINK-DETECTION-WITH-OPENCV-AND-DLIB/tree/master)

# 涉及的步骤
1.定位图像中的人脸

2.提取眼睛的位置

3.计算 EAR（眼睛纵横比）

4.对 EAR 进行阈值处理以确定此人是否在眨眼

# EAR （Eye Aspect Ratio） 公式如下：
$$
EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||}
$$

# 结果

![Example output](result.gif)

