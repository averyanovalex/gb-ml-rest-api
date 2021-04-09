# python-flask-docker ml-res-api
Итоговый проект курса "Машинное обучение в бизнесе"

Цель:

Научиться и продемонстрировать навыки развертывания обученной модели в виде простого rest-api на flask

Стек:

ML: sklearn, pandas, numpy
API: flask
Данные: с kaggle - https://www.kaggle.com/blackmoon/russian-language-toxic-comments

Задача: определение токсичного комментария. Бинарная классификация

Используемые признаки:
- comment (text)

Преобразования признаков: tfidf

Модель: logreg

### Структура репозитория:
/app - приложение с моделью (api-server на flask, порт 8180)
/app/front - простое front приложение (flask, port 8181)
/data_example - пример датасета на котором проводилось обучение модели
/jupiter - тетрадка с обучением и оценкой качества модели

### Формат api:
Пример post-запроса: http://localhost:8180/predict
```
{"comment": "мне очень понравился твой комментарий  классный"}
```
Пример ответа:
 - prediction: предсказанный класс (0 - не токсичный, 1 - токсичный)
 - prediction_probability: вероятность предсказания (predict_proba)
 - probability_threshold: порог вероятности предсказания, заложенный в модель
 - success: сервис успешно выполнил предсказание (true) или были ошибки (false)
```
{
    "prediction": 0,	
    "prediction_probability": 0.25592150626701377,
    "probability_threshold": 0.3679905070066962,
    "success": true
}
```


### Как собрать и развернуть приложение

#### Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/averyanovalex/gb-ml-rest-api
```
#### Обучаем и сохраняем модель
Скачиваем датасет (по ссылке выше или свой)
Обучем в тетрадке /jupiter/training_model.ipynb, сохраняем обученную модель (этот каталог нужно будет указать позже).
Имя файла: logreg_model.dill

#### Собираем Docker контейнер
Переходим в каталог репозитория и вызываем команду docker build
```
$ cd gb-ml-rest-api
$ docker build -t avea/gb_docker_ml_flask_rest_api .
```

#### Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -p 8181:8181 -v <your_local_path_to_pretrained_models>:/app/app/models avea/gb_docker_ml_flask_rest_api
```

#### Переходим на localhost:8181 или отправляем post-запросы на localhost:8180
