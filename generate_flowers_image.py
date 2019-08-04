''' Code to generate the image composed of 4 flower images used in the
jupyter notebook "flowers.ipynb".'''

import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

image_folder = "sample_images"

fig, (ax2, ax1, ax4, ax3) = plt.subplots(ncols=4, figsize=(12,4))

ax1.imshow(Image.open(os.path.join(image_folder, "buttercup.jpg")))
ax2.imshow(Image.open(os.path.join(image_folder, "bird_of paradise.jpg")))
ax3.imshow(Image.open(os.path.join(image_folder, "moon_orchid.jpg")))
ax4.imshow(Image.open(os.path.join(image_folder, "corn_poppy.jpg")))
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")
plt.savefig(os.path.join("four_flowers.jpg"), dpi=300)