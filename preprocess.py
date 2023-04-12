import cv2
import numpy as np
from scipy.signal import convolve2d

def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def remove_wavy_noise(img_gray, low_pass_filter_size=25):
    # 执行快速傅里叶变换并计算频谱幅度
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 设计低通滤波器
    rows, cols = img_gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - low_pass_filter_size:crow + low_pass_filter_size,
         ccol - low_pass_filter_size:ccol + low_pass_filter_size] = 1

    # 应用低通滤波器
    fshift_filtered = fshift * mask
    magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered))

    # 执行逆快速傅里叶变换
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

def blind_deconvolution(img, psf, iterations=10):
    deconvolved = np.full(img.shape, 0.5)
    for _ in range(iterations):
        reconvolved = convolve2d(deconvolved, psf, 'same')
        relative_blur = img / (reconvolved + 1e-10)
        deconvolved *= convolve2d(relative_blur, psf[::-1, ::-1], 'same')
    return deconvolved

# 读取图像
input_image = 'demo.jpg'
img = cv2.imread(input_image, cv2.IMREAD_COLOR)

# 应用反锐化掩膜
img = unsharp_mask(img)

# 将BGR图像分解为单独的通道
b, g, r = cv2.split(img)

# 应用盲去卷积
psf = np.ones((5, 5)) / 25
b_restored = blind_deconvolution(b, psf)
# b_restored = remove_wavy_noise(b_restored)
g_restored = blind_deconvolution(g, psf)
# g_restored = remove_wavy_noise(g_restored)
r_restored = blind_deconvolution(r, psf)
# r_restored = remove_wavy_noise(r_restored)

# 将图像数据类型转换回uint8并裁剪到0-255范围
b_restored = np.clip(b_restored, 0, 255).astype('uint8')
g_restored = np.clip(g_restored, 0, 255).astype('uint8')
r_restored = np.clip(r_restored, 0, 255).astype('uint8')

# 重新组合通道以获得恢复后的BGR图像
img_restored = cv2.merge((b_restored, g_restored, r_restored))
# filtered_img = cv2.bilateralFilter(img_restored, d=9, sigmaColor=75, sigmaSpace=75)
filtered_img = cv2.fastNlMeansDenoisingColored(img_restored, None, 10, 10, 7, 21)

# 显示图像
print('Original image')
cv2.imwrite('./demo_restored_sharpen.jpg', img_restored)
print('Restored image')