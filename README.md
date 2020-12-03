# Улучшение качества изображений с помощью технологии SRGAN

![Alt text](/Third-party/srgan_example.jpg)

Данный проект представляет собой нейронную сеть состоящую из трех моделей: Generator, Discriminator и VGG19 ( для оценки потерь).

В файле Models.py находятся классы для построения данных трех моделей с соотвествующей архитектурой. Файл DataLoader.py представляет класс для работы с датасетом и предварительной подготовки тренировочных данных. Основной скрипт для создания моделей и их обучения содержится в файле main.py

Обучающий скрипт предоставляет полную найстроку параметров нейронной сети, датасетов и процесса обучения. Для просмотра возможных конфигураций пропишите:

```sh
$ python main.py --help
```
Для запуска обучения необходимо прописать:

```sh
$ python main.py --[Параметры обучения]
```

Все дополнительные материалы, которые могут помочь в понимании архитекутры, в настройке обучения находятся в папке **Third-party**.
Обучение проводилось на DIV2k датасете, который можно скачать по ссылке:
[Alt text](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
Также вы можете найти датасет, который использовался в обучении, все картинки уже размера 512*512. Процесс обучения настроен таким образом, что изображения низкого качества не требуются, они получается в результате понижения качества исходных картинок. 
Обучение проводилось в облачном сервисе [Alt text](https://colab.research.google.com/ "Google Collaboratory"), который предсталяет мощности для быстрого обучения моделей. Файл, запускаемый в данном сервисе, называется *srgan_v1.ipynb*. 
Также для понимания архитектуры самой сети и ее принципа обучения представлена научная статья в формате pdf.
