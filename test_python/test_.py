import pybicos
import cv2 as cv
import matplotlib.pyplot as plt
from time import perf_counter 
lstack = [cv.imread(f"data/left/{i}_left.png", cv.IMREAD_UNCHANGED) for i in range(30)]
rstack = [cv.imread(f"data/right/{i}_right.png", cv.IMREAD_UNCHANGED) for i in range(30)]

# resize to half size
# lstack = [cv.resize(img, (0, 0), fx=0.5, fy=0.5) for img in lstack]
# rstack = [cv.resize(img, (0, 0), fx=0.5, fy=0.5) for img in rstack]

cfg = pybicos.Config()
cfg.nxcorr_threshold = 0.9
# cfg.subpixel_step = 0.1


disparity, correlation_map = pybicos.match(lstack, rstack, cfg)

start = perf_counter()
disparity, correlation_map = pybicos.match(lstack, rstack, cfg)
end = perf_counter()
print(f"Time taken: {end - start} seconds")

cv.imwrite('disparity.tiff', disparity)
cv.imwrite('correlation_map.tiff', correlation_map)
plt.imshow(disparity, cmap='jet')
plt.colorbar()
plt.savefig('disparity.png')

plt.imshow(correlation_map, cmap='jet')
plt.savefig('correlation_map.png')