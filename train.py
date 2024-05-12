from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255) #preprocessing images before feeding them into the model.
val_datagen = ImageDataGenerator(rescale=1./255)  # The ImageDataGenerator class is configured to rescale the pixel values of images to a range between 0 and 1.

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (48,48),  #dimensions 48x48 pixels
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)

emotion_model = Sequential() #This line initializes a Sequential model, which is a linear stack of layers. We'll add layers to this model one by one.
#img processing -> feature extraction -> dropout
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (48,48,1))) # 32 filters of size 3X3 48*48 px with 1 channel 
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))# MaxPooling2D: Pooling layer for downsampling and reducing the spatial dimensions of feature maps.
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())# Flatten: Layer to convert 2D feature maps into a 1D feature vector for input to dense layers.
emotion_model.add(Dense(1024, activation='relu'))#simple layer of neurons in which each neuron receives input from all the neurons of previous layer, thus called as dense. Dense Layer is used to classify image based on output from convolutional layers
emotion_model.add(Dropout(0.5))# Dropout: Layer for preventing overfitting by randomly dropping a fraction of neurons during training.
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy']) #algorithm that dynamically adjusts learning rates for each parameter during training, enhancing convergence speed and performance.

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch = 28709 // 64,
    epochs=75,
    validation_data = val_generator,
    validation_steps = 7178 // 64
)

emotion_model.save_weights('model.h5')