import cv2
import numpy as np
from scipy.signal import convolve2d

def blind_deconvolution(img, psf, iterations=50):
    deconvolved = np.full(img.shape, 0.5)
    for _ in range(iterations):
        reconvolved = convolve2d(deconvolved, psf, 'same')
        relative_blur = img / (reconvolved + 1e-10)
        deconvolved *= convolve2d(relative_blur, psf[::-1, ::-1], 'same')
    return deconvolved

# 读取图像
input_image = 'demo.jpg'
img = cv2.imread(input_image, cv2.IMREAD_COLOR)

# 将BGR图像分解为单独的通道
b, g, r = cv2.split(img)

# 应用盲去卷积
psf = np.ones((5, 5)) / 25
b_restored = blind_deconvolution(b, psf)
g_restored = blind_deconvolution(g, psf)
r_restored = blind_deconvolution(r, psf)

# 将图像数据类型转换回uint8并裁剪到0-255范围
b_restored = np.clip(b_restored, 0, 255).astype('uint8')
g_restored = np.clip(g_restored, 0, 255).astype('uint8')
r_restored = np.clip(r_restored, 0, 255).astype('uint8')

# 重新组合通道以获得恢复后的BGR图像
img_restored = cv2.merge((b_restored, g_restored, r_restored))

# 显示图像
cv2.imwrite('demo_restored.jpg', img_restored)