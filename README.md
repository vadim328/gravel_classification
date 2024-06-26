# crushed_stone_classification

### Цель проекта

Автоматизированная система предназначенная для определения фракции щебня в кузове грузовика на изображении

### Описание

Система состоит из двух Docker-контейнеров в которых развернуты модели обнаружения объектов и классификации. В качестве модели детекции используется - YOLOv8s, а для классификации - Deit3-small.

Обученная модель детектора YOLOv8s на Hugging Face - https://huggingface.co/Vades/GravelDetectionYOLOv8/tree/main

Обученные модели классификаторов на Hugging Face - [https://huggingface.co/Vades/GravelDetectionYOLOv8/tree/main](https://huggingface.co/Vades/GravelClassificationModels/tree/main)

Для взаимодействия с системой реализован POST метод /pred. В качестве входных параметров данный метод может принимать: изображение для классификации, алгоритм - одношаговый или двухшаговый и формат возвращаемого результата - json объект или размеченное изображение.

Привер запроса: 

files = {'file': open('image.jpeg', 'rb')}

params = {'algorithm': 'one_stage', 'return_format': 'image'}

response = requests.post('http://178.154.221.81:8000/pred', files=files, data=params)

Схема двухшагового процесса:

![image](https://github.com/vadim328/gravel_classification/assets/28571240/4f6ba4ca-5f29-4f83-8b49-cc147e973077)

Пример работы системы:

![image](https://github.com/vadim328/gravel_classification/assets/28571240/56ce36e1-694a-4bfb-bf0a-e76c7b139fb9)



