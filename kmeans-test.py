from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image_path = '/Users/ahnwooseok/Downloads/github/SWAdvancedProject/happy_couple.jpg'
image = Image.open(image_path)

image_np = np.array(image)
h, w, c = image_np.shape
image_reshaped = image_np.reshape(-1, c)

n_colors = 10
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
kmeans.fit(image_reshaped)

labels = kmeans.predict(image_reshaped)
labeled_image = labels.reshape(h, w)

palette = kmeans.cluster_centers_.astype(int)

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.title('Labeled Image')
plt.axis('off')

plt.subplot(1, 2, 2)
# plt.imshow([palette], shape=(1, n_colors, c))
plt.imshow([palette])
plt.title('Palette')
plt.axis('off')

plt.tight_layout()
plt.show()

labeled_image_path = 'labeled_image.png'
palette_image_path = 'palette.png'
Image.fromarray(labeled_image).save(labeled_image_path)
Image.fromarray(palette.reshape(1, n_colors, c)).save(palette_image_path)

(labeled_image_path, palette_image_path)
