import tensorflow as tf
from tf_keras import datasets,layers,models 
import matplotlib.pyplot as mlt
(train_image,train_labels),(test_image,test_label) = datasets.mnist.load_data()
train_image = train_image/255.0
test_image = test_image/255.0

model = models.Sequential(
    [
        layers.Flatten(input_shape = (28,28)),#input layer
        layers.Dense(128,activation = 'relu'),
        layers.Dense(10,activation= 'softmax')
    ]
)
model.compile(
    optimizer='adam',
    loss= 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(train_image,train_labels)
mlt.imshow(test_image[0],cmap=mlt.cm.binary)
mlt.show()