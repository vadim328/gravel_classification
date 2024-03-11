from PIL import Image
import os
import json
import re
import yaml
import uuid


with open('config.yaml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)


def get_coordinates(title) -> tuple[int, int, int, int]:
    """
    Возвращает координаты верхнего левого и нижнего правого углов.
            Параметры:
                    title (string): наименование файла
            Возвращаемое значение:
                    tuple (int, int, int, int): x и y координаты двух точек
    """
    exp = '.json'
    f = open(config['path_label'] + title + exp,)
    data = json.load(f)

    x1 = int(data['shapes'][0]['points'][0][0])
    y1 = int(data['shapes'][0]['points'][0][1])
    x2 = int(data['shapes'][0]['points'][1][0])
    y2 = int(data['shapes'][0]['points'][1][1])

    print(x1, y1, x2, y2)

    # Closing file
    f.close()
    return x1, y1, x2, y2


def create_id() -> list[str]:
    """
        Возвращает уникальные идентификаторы для сохранения файлов.
                Возвращаемое значение:
                        list_id (list): список идентификаторов
        """
    list_id = []
    for i in range(0, 4):
        list_id.append(str(uuid.uuid4())[:8])
    return list_id


path_img = config['path_image']
for file_name in os.listdir(path_img):
    if file_name.endswith('.jpeg'):
        img = Image.open(path_img + file_name)
        last_dot = file_name.rfind('.')
        x1, y1, x2, y2 = get_coordinates(file_name[:last_dot])

        img_crop = img.crop((x1, y1, x2, y2))

        width, height = img_crop.size
        print(width, height)
        s: int = config["center_square_size_big"]
        img_center = img_crop.crop(((width - s) // 2,
                                    (height - s) // 2,
                                    (width + s) // 2,
                                    (height + s) // 2))

        width, height = img_center.size
        print(width, height)

        # Разделение на 4 части
        part_width = width // 2
        part_height = height // 2

        # Координаты углов каждой части
        part1_coords = (0, 0, part_width, part_height)
        part2_coords = (part_width, 0, width, part_height)
        part3_coords = (0, part_height, part_width, height)
        part4_coords = (part_width, part_height, width, height)

        # Вырезаем каждую часть
        part1 = img_center.crop(part1_coords)
        part2 = img_center.crop(part2_coords)
        part3 = img_center.crop(part3_coords)
        part4 = img_center.crop(part4_coords)

        pattern = r'_\d{1,2}-\d{1,3}_'
        fraction = re.findall(pattern, file_name)[0][1:-1]
        lst_id = create_id()
        part1.save('../dataset for classification/' + fraction + '/' + lst_id[0] +
                   '_' + fraction + '.jpeg', quality=95)
        part2.save('../dataset for classification/' + fraction + '/' + lst_id[1] +
                   '_' + fraction + '.jpeg', quality=95)
        part3.save('../dataset for classification/' + fraction + '/' + lst_id[2] +
                   '_' + fraction + '.jpeg', quality=95)
        part4.save('../dataset for classification/' + fraction + '/' + lst_id[3] +
                   '_' + fraction + '.jpeg', quality=95)
