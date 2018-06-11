### ResizeMethod
类：</br>
class ResizeMethod

功能：</br>
1. adjust_brightness(...): 调整RGB图像或灰度图的亮度。
1. adjust_contrast(...): 调整RGB图像或灰度图的对比度。
1. adjust_gamma(...): 在输入图像上执行伽玛校正。
1. adjust_hue(...): 调整RGB图像的色调。
1. adjust_saturation(...): 调整RGB图像的饱和度。
1. central_crop(...): 从图像的中央区域裁剪图像。
1. convert_image_dtype(...): 将图像转换为dtype，如果需要，缩放其值。
1. crop_and_resize(...): 对输入图像做剪裁并通过插值方法调整尺寸。
1. crop_to_bounding_box(...): 指定边界的裁剪图像。
1. decode_gif(...): 将GIF编码图像的第一帧解码为 uint8 tensor。
1. decode_image(...): 图像解码操作，包含了 decode_gif, decode_jpeg,和 decode_png。
1. decode_jpeg(...): 将jpeg编码图像解码为 uint8 tensor。
1. decode_png(...): 将png编码图像解码为 uint16 tensor。
1. draw_bounding_boxes(...): 在一个batch的图像上绘制边框。
1. encode_jpeg(...): JPEG图像编码。
1. encode_png(...): PNG图像编码。
1. extract_glimpse(...): 从指定的位置提取指定尺寸的区域，如果超过了原图像的尺寸，将随机填充。
1. flip_left_right(...): 水平翻转图像 。
1. flip_up_down(...): 上下翻转图像。
1. grayscale_to_rgb(...): 单个或多个图像灰度转RGB。
1. hsv_to_rgb(...): 单个或多个图像HSV转RGB。
1. non_max_suppression(...): 根据分数降序选择边界框，分数是一个输入，函数别没有计算分数的规则，其实只是提供了一种降序选择操作。
1. pad_to_bounding_box(...): 补零，将图像填充到指定的宽高。
1. per_image_standardization(...): 图像标准化（不是归一化）。
1. random_brightness(...): 通过随机因子调整图像的亮度。
1. random_contrast(...): 通过随机因子调整图像的对比度。
1. random_flip_left_right(...): 随机水平翻转图像。
1. random_flip_up_down(...): 随机上下翻转图像。
1. random_hue(...): 通过随机因子调整RGB图像的色调。
1. random_saturation(...):通过随机因子调整RGB图像的饱和度。
1. resize_area(...): 应用区域插值调整图像尺寸。
1. resize_bicubic(...): 应用双三次插值调整图像尺寸。
1. resize_bilinear(...): 应用双线性内插值调整图像尺寸。
1. resize_nearest_neighbor(...): 应用最邻近插值调整图像尺寸。
1. resize_images(...): 使用指定的方法调整图像尺寸（其实包含的是上面四种插值方法）。
1. resize_image_with_crop_or_pad(...): 根据目标图像的宽高（自动）裁剪或填充图像。
1. rgb_to_grayscale(...): 单个或多个图像RGB转灰度图。
1. rgb_to_hsv(...): 单个或多个图像RGB转HSV。
1. rot90(...): 将图像逆时针旋转90度。
1. sample_distorted_bounding_box(...): 为图像生成单个随机变形的边界框。
1. total_variation(...): 计算一个图像或多个图像的总体变动（输入图像中相邻像素值的绝对差异） 
1. transpose_image(...): 交换图像的第一维和第二维（输入要求是3D，没有batch，也就是宽和高的变换）

### resize
```
tf.image.resize_images(images, new_height, new_width, method=0)
```
tf.image.resize_images函数中method参数取值与相对应的图像大小调整算法：

![resize_image_1.png](https://i.imgur.com/w9EwJPy.png)