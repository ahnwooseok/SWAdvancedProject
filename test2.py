from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import KMeans

def process_image(image_path, n_clusters=10):

    image = Image.open(image_path)
    image = image.resize((image.width // 3, image.height // 3))
    image_np = np.array(image)

    pixels = image_np.reshape(-1, image_np.shape[2])

    model = KMeans(n_clusters=n_clusters)
    model.fit(pixels)
    palette = model.cluster_centers_.astype(int)
    labels = model.labels_

    new_pixels = np.array([palette[label] for label in labels])
    new_image_np = new_pixels.reshape(image_np.shape)

    new_image = Image.fromarray(new_image_np.astype('uint8'), 'RGB')

    return new_image, palette, labels, image.size

n_clusters = 10
new_image, palette, labels, original_size = process_image('/Users/ahnwooseok/Downloads/github/SWAdvancedProject/happy_couple.jpg', n_clusters)

labeled_image = Image.new('RGB', original_size)
draw = ImageDraw.Draw(labeled_image)
font = ImageFont.load_default()
font_size = 10
for i in range(0, labeled_image.height, font_size):
    for j in range(0, labeled_image.width, font_size):
        label = labels[(i // 3) * (labeled_image.width // 3) + (j // 3)]
        draw.text((j, i), str(label), font=font, fill=str(palette[label]))
        # draw.text((j, i), str(label), font=font, align="center")
        # draw.text((j, i), str(label), font=font, align="left")

labeled_image.save('path_to_save_labeled_image.jpg')
