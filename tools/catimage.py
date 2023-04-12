# import cv2
# import numpy as np

original_image_path = "/mnt/data_ssd1/lzp/LargeScaleNeRFPytorch/logs/360/bonsai_nov29_3TruesobelEdgeWeight0.15/render_test_heatmap/0.png"
render_image_path = "/mnt/data_ssd1/lzp/LargeScaleNeRFPytorch/logs/360/bonsai_nov29_3TruesobelEdgeWeight0.15/render_test_heatmap/gt_0.png"
original_image_with_heatmap_path = "/mnt/data_ssd1/lzp/LargeScaleNeRFPytorch/logs/360/bonsai_nov29_3TruesobelEdgeWeight0.15/render_test_heatmap/0_heat.png"
render_image_with_heatmap_path = "/mnt/data_ssd1/lzp/LargeScaleNeRFPytorch/logs/360/bonsai_nov29_3TruesobelEdgeWeight0.15/render_test_heatmap/gt_0_heat.png"

# original_image = cv2.imread(original_image_path)
# render_image = cv2.imread(render_image_path)
# original_image_with_heatmap = cv2.imread(original_image_with_heatmap_path)
# render_image_with_heatmap = cv2.imread(render_image_with_heatmap_path)


# cv2.imwrite("cat_image_false.png", cat_image)

from PIL import Image
import numpy as np

img1 = Image.open(original_image_path)
img2 = Image.open(render_image_path)
img3 = Image.open(original_image_with_heatmap_path)
img4 = Image.open(render_image_with_heatmap_path)

width, height = img1.size
result_width = width * 2 + 10
result_height = height
result = Image.new("RGB", (result_width, result_height), (255, 255, 255))

result.paste(img1, (0, 0))
result.paste(img2, (width + 5, 0))
result.paste(img3, (width * 2 + 5, 0))
result.paste(img4, (width * 3 + 10, 0))

result.save("result.png")
