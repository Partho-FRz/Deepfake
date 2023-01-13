import numpy as np
from classifiers import *
from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# classifier = model_DF()
classifier = model_FF()



classifier.load('./weights/Model_FF.h5')


dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        # batch_size=2,
        class_mode='binary',
        subset='training')


X, y = generator.next()


print('Predicted :', classifier.predict(X), '\nReal class :', y)





predictions = compute_accuracy(classifier, 'test_videos')
for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])