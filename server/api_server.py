from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import requests
import os
import cv2
import yaml
from edit_image import EditImage


with open('config.yaml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Status OK"}


detection_server = config['detection_server']
classification_server = config['classification_server']


@app.post('/pred/')
async def upload_image(file: UploadFile = File(...),
                       algorithm: str = Form('one_stage'),
                       return_format: str = Form('json')):
    image = await file.read() # считаываем изображение
    response_detection = requests.put(detection_server, data=image) # запрос на обнаружение

    image = EditImage(image, response_detection.json()[0])
    image.draw_frame() # Добавляем рамку на изображение

    #return {"status": "OK"}

    if algorithm == 'one_stage':
        image.write_class(response_detection.json()[0]['class_name'])
    elif algorithm == 'two_stage':
        center_image = image.get_center_img() # Получаем центр изображения
        response_classification = requests.put(classification_server, data=center_image) # запрос на классификацию
        image.write_class(response_classification.json()['pred_class'])

    # Сохраняем изображение
    filename = file.filename
    file_path = os.path.join("images", filename)
    cv2.imwrite(file_path, image.get_image())

    if return_format == 'image':
        return FileResponse(file_path)

    return {'fraction': image.get_fraction()}
