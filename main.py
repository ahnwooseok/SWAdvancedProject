from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def process_image(image_path, n_clusters=10):

    # 이미지 로드
    image = Image.open(image_path)

    # 이미지 리 사이징
    # image = image.resize((image.width // 3, image.height // 3))


    # 이미지를 numpy 배열, 픽셀 단위 변환
    image_np = np.array(image)
    pixels = image_np.reshape(-1, image_np.shape[2])

    # K-Mean 클러스터링으로 색깔 군집화, RGB -> 같이 묶기
    # 팔레트 색깔로 군집화
    # 색깔 개수 : n_clusters
    model = KMeans(n_clusters = n_clusters)
    model.fit(pixels)
    palette = model.cluster_centers_.astype(int)
    labels = model.labels_
    new_pixels = np.array([palette[label] for label in labels])
    new_image_np = new_pixels.reshape(image_np.shape)
    new_image = Image.fromarray(new_image_np.astype('uint8'), 'RGB')

    return new_image, palette

# 초기 할당 값
n_clusters = 10
# new_image, palette = process_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/happy_couple.jpg', n_clusters)
# new_image, palette = process_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/bridge.png', n_clusters)
# new_image, palette = process_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/bridge_sea.png', n_clusters)
# new_image, palette = process_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/forest.png', n_clusters)
# new_image, palette = process_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/peter.jpeg', n_clusters)
new_image, palette = process_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/dog.jpg', n_clusters)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(new_image)
plt.title('Processed Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow([palette], aspect='auto')
plt.title('Palette')
plt.axis('off')

plt.show()
