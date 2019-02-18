import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab
from PIL import Image
import os


fig = plt.figure(figsize=(10, 3))
os.chdir('../../Report/imgs/raw_imgs')
# lab = Image.open('n02100735_4582_128.png')
# lab= Image.open('n02509815_3046_128_real.png')
# rgb= Image.open('n02091134_1006_128.png')
rgb= Image.open('n02125311_38943_128.png')
lab = rgb2lab(rgb)
image = np.asarray(lab)

ax = fig.add_subplot(1, 3, 1)
z = np.zeros(image.shape)
z[:, :, 0] = image[:, :, 0]
L = lab2rgb(z)
ax.imshow(L)
ax.axis("off")
ax.set_title("L: lightness")

ax = fig.add_subplot(1, 3, 2)
A = np.zeros(image.shape)
A[:, :, 0] = 40
A[:, :, 1] = image[:, :, 1]
A_RGB=lab2rgb(A)
ax.imshow(A_RGB);
ax.axis("off")
ax.set_title("A: color spectrums green to red")

ax = fig.add_subplot(1, 3, 3)
B = np.zeros(image.shape)
B[:, :, 0] = 40
B[:, :, 2] = image[:, :, 2]
B_RGB=lab2rgb(B)
ax.imshow(B_RGB)
ax.axis("off")
ax.set_title("B: color spectrums blue to yellow")
plt.show()


figAB = plt.figure(figsize=(10, 3))

axAB = figAB.add_subplot(1, 3, 1)
axAB.imshow(rgb)
axAB.axis("off")
axAB.set_title("RGB")


axAB = figAB.add_subplot(1, 3, 2)
axAB.imshow(L)
axAB.axis("off")
axAB.set_title("L: lightness")

axAB = figAB.add_subplot(1, 3, 3)
AB = A
AB[:, :, 2] = B[:, :, 2]
AB=lab2rgb(AB)
axAB.imshow(AB)
axAB.axis("off")
axAB.set_title("AB Color")

plt.show()