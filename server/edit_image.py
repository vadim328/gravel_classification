import yaml
import numpy as np
import cv2


with open('config.yaml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)


class EditImage:
    """Класс для обработки изображения"""
    def __init__(self, image, detected_data, fraction=None):
        self.image = self.convert_to_cv2(image)
        self.x1, self.y1, self.x2, self.y2 = self.parsing_coordinates(detected_data)
        self.fraction = fraction

    def convert_to_cv2(self, image):
        image_np = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        return image

    def parsing_coordinates(self, detected_data) -> tuple[int, int, int, int]:
        """
        Возвращает координаты верхнего левого и нижнего правого углов.
                Параметры:
                        detected_data (map): json объект
                Возвращаемое значение:
                        tuple (int, int, int, int): x и y координаты двух точек
        """

        x1 = int(detected_data['bbox'][0])
        y1 = int(detected_data['bbox'][1])
        x2 = int(detected_data['bbox'][2])
        y2 = int(detected_data['bbox'][3])
        # print(x1, y1, x2, y2)

        return x1, y1, x2, y2

    def get_center_img(self):
        # Координаты прямоугольника (левый верхний угол и правый нижний угол)
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        img_crop = self.image[y1:y2, x1:x2]

        # Получаем размер изображения
        height, width = img_crop.shape[:2]
        s: int = config['center_square_size']

        # Вычисляем координаты верхнего левого угла для обрезки
        start_x = (width - s) // 2
        start_y = (height - s) // 2

        # Обрезаем изображение по центру
        img_center = img_crop[start_y:start_y + s, start_x:start_x + s]

        # Конвертируем в байты для отправки
        img_center_bytes = cv2.imencode('.jpeg', img_center)[1].tobytes()

        return img_center_bytes

    def draw_frame(self):
        # Координаты прямоугольника (левый верхний угол и правый нижний угол)
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        image = self.image
        color = (0, 255, 0)  # Цвет рамки
        thickness = 10  # Толшина рамки

        # Рисуем прямоугольник на изображении
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        self.image = image

    def write_class(self, data, algorithm):
        image = self.image
        if algorithm == 'one_stage':
            self.fraction = str(data['class_label'])
        elif algorithm == 'two_stage':
            self.fraction = str(data['pred_class'])

        # Параметры для putText
        org = (self.x1, self.y1)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        color = (0, 0, 0)
        thickness = 4

        cv2.putText(image, self.fraction, org, font_face, font_scale, color, thickness)
        self.image = image

    def get_image(self):
        return self.image

    def get_fraction(self):
        return self.fraction
