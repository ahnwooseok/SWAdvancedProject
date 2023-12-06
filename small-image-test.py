from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def label_image(image_path, n_colors=10):

    image = Image.open(image_path)
    image = image.convert('RGB')



    image_np = np.array(image)
    # image_np = np.array(image_path)
    
    height, width, c = image_np.shape
    image_reshaped = image_np.reshape(-1, c)

    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(image_reshaped)
    labels = kmeans.predict(image_reshaped)
    labeled_image = labels.reshape((height, width))
    return labeled_image, kmeans.cluster_centers_

n_colors = 10  # cluster 군집 개수
labeled_image, centers = label_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/happy_couple.jpg', n_colors=n_colors)
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.colorbar()
plt.axis('off')
plt.show()
