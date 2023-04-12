import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import roberts, prewitt

image = cv2.imread('/mnt/data_ssd1/lzp/LargeScaleNeRFPytorch/data/360_v2/bonsai/images/DSCF5583.JPG', cv2.IMREAD_GRAYSCALE)

edges_gray_image = cv2.Canny(image, 10, 180)
# _, edges_gray_image = cv2.threshold(edges_gray_image, 40, 255, cv2.THRESH_BINARY)
canny_edges = edges_gray_image.astype(np.float32) / 255.
# canny_edges = cv2.Canny(image, 100, 200)
# canny_edges = cv2.convertScaleAbs(canny_edges)
# canny_edges = cv2.dilate(canny_edges, None)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = np.hypot(sobel_x, sobel_y)
_, edges_gray_image = cv2.threshold(sobel_edges, 70, 255, cv2.THRESH_BINARY)
sobel_edges = edges_gray_image.astype(np.float32) / 255.

# sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
# sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
# sobel_edges = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
# sobel_edges = cv2.convertScaleAbs(sobel_edges)
# sobel_edges = cv2.dilate(sobel_edges, None)

laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)
_, edges_gray_image = cv2.threshold(laplacian_edges, 7, 255, cv2.THRESH_BINARY)
laplacian_edges = edges_gray_image.astype(np.float32) / 255.
# laplacian_edges = cv2.Laplacian(image, cv2.CV_64F)
# laplacian_edges = cv2.convertScaleAbs(laplacian_edges)
# laplacian_edges = cv2.dilate(laplacian_edges, None)

roberts_kernel_x = np.array([[1, 0], [0, -1]])
roberts_kernel_y = np.array([[0, 1], [-1, 0]])

roberts_x = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_x)
roberts_y = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_y)

roberts_edges = roberts(image)
_, edges_gray_image = cv2.threshold(roberts_edges, 0.05, 255, cv2.THRESH_BINARY)
roberts_edges = edges_gray_image.astype(np.float32) / 255.
# roberts_edges = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)
# roberts_edges = cv2.convertScaleAbs(roberts_edges)
# roberts_edges = cv2.dilate(roberts_edges, None)


prewitt_edges = prewitt(image)
_, edges_gray_image = cv2.threshold(prewitt_edges, 0.05, 255, cv2.THRESH_BINARY)    
prewitt_edges = edges_gray_image.astype(np.float32) / 255.
# prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
# prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
# prewitt_x = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_x)
# prewitt_y = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_y)
# prewitt_edges = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
# prewitt_edges = cv2.convertScaleAbs(prewitt_edges)
# prewitt_edges = cv2.dilate(prewitt_edges, None)

# titles = ['Original Image', 'Canny', 'Sobel', 'Laplacian', 'Roberts', 'Prewitt']
images = [image, canny_edges, sobel_edges, laplacian_edges, roberts_edges, prewitt_edges]
titles = ['(a) Original View Image',
          '(b) Canny Edge Detection',
          '(c) Sobel Edge Detection',
          '(d) Laplacian Edge Detection',
          '(e) Roberts Edge Detection',
          '(f) Prewitt Edge Detection']

plt.figure(figsize=(15, 8))

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.annotate(titles[i], fontsize=14, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='center')

plt.tight_layout()
plt.savefig('./figs/edge_detection.png', dpi=300, bbox_inches='tight')
