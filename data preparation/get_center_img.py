from PIL import Image
import yaml


with open('config.yaml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

path = './detected objects/images/bffb53dc-52_ЩПС_фр._0-20_мм_НЕГОСТ.jpeg'

img = Image.open(path)

x1, y1, x2, y2 = 0, 826, 305, 1941
img_crop = img.crop((x1, y1, x2, y2))

width, height = img_crop.size
print(width, height)
s: int = config["center_square_size"]
img_center = img_crop.crop(((width - s) // 2,
                            (height - s) // 2,
                            (width + s) // 2,
                            (height + s) // 2))

img_center.show()
