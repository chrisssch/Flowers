import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.pyplot import imshow

fig, (ax2, ax1, ax4, ax3) = plt.subplots(ncols=4, figsize=(12,4))
ax1.imshow(Image.open("sample_images/buttercup.jpg"))
ax2.imshow(Image.open("sample_images/bird of paradise.jpg"))
ax3.imshow(Image.open("sample_images/moon orchid.jpg"))
ax4.imshow(Image.open("sample_images/corn poppy.jpg"))
ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")

plt.savefig("sample_images/four_flowers.jpg", dpi=300)
