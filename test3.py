from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans
from skimage.measure import find_contours

def process_image_with_contours(image_path, n_clusters=10):

    image = Image.open(image_path)
    image.thumbnail((200, 200))
    image_np = np.array(image)

    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_.reshape(image_np.shape[:2])

    labeled_image = Image.new('RGB', image_np.shape[:2][::-1], (255, 255, 255))
    draw = ImageDraw.Draw(labeled_image)
    font = ImageFont.load_default()

    for i in range(n_clusters):
        contours = find_contours(labels == i, 0.5)
        for contour in contours:
            contour = contour * 3
            contour = [(int(x), int(y)) for x, y in contour]
            draw.line(contour, fill=(0, 0, 0), width=2)

        indices = np.where(labels == i)
        centroid = np.mean(indices, axis=1).astype(int)

        draw.text((centroid[1]*3, centroid[0]*3), str(i), fill=(0, 0, 0), font=font)

    return labeled_image

labeled_image = process_image_with_contours('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/happy_couple.jpg', n_clusters=10)


labeled_image_path = 'happy_couple_labeled_with_contours.jpg'
labeled_image.save(labeled_image_path)
labeled_image.show()

labeled_image_path
