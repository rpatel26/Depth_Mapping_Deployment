from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

image_file = Image.open('depth_image2.png')
width, height = image_file.size

image = np.asarray(image_file)
heat_map_image = image[:,:,0:3]

print(type(image_file))
print("height = ", height)
print("width = ", width)
print("shpae = ", np.shape(image))

count = 0
for i in image:
	count += 1

print('count = ', count)

plt.imshow(heat_map_image)
plt.show()

plt.imshow(image)
plt.show()