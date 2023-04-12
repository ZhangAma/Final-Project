import cv2
import numpy as np

color_image = cv2.imread('/mnt/data_ssd1/lzp/LargeScaleNeRFPytorch/data/tanks_and_temples/tat_intermediate_Playground/train/rgb/00001.png')
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
edges_gray_image = cv2.Canny(gray_image, 252, 255)
edges_color_image = cv2.cvtColor(edges_gray_image, cv2.COLOR_GRAY2BGR)
enhanced_color_image = cv2.addWeighted(color_image, 0.85, edges_color_image, 0.15, 0)

# cv2.imshow('Original Color Image', color_image)
# cv2.imshow('Enhanced Edges Color Image', enhanced_color_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('demo_edge.png', edges_color_image)
cv2.imwrite('demo.png', enhanced_color_image)