import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# Load and preprocess data
def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels

# Build model function (parameterized for MEC adjustments)
def build_model(use_mec_adjustments=False):
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)

    if use_mec_adjustments:
        x = Dropout(0.2)(x)  # MEC Adjustment: Dropout

    x = Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=l2(0.001) if use_mec_adjustments else None)(x)  # Conditional L2 regularization
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=l2(0.001) if use_mec_adjustments else None)(x)  # Conditional L2 regularization

    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate model function (parameterized for data augmentation)
def train_and_evaluate(model, train_images, train_labels, test_images, test_labels, use_augmentation=False):
    if use_augmentation:
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
        history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                            epochs=10, validation_data=(test_images, test_labels))
    else:
        history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    return test_acc

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()

    # Without MEC Adjustments
    model_without_mec = build_model(use_mec_adjustments=False)
    accuracy_without_mec = train_and_evaluate(model_without_mec, train_images, train_labels, test_images, test_labels, use_augmentation=False)
    print(f'Accuracy without MEC Adjustments: {accuracy_without_mec:.4f}')

    # With MEC Adjustments
    model_with_mec = build_model(use_mec_adjustments=True)
    accuracy_with_mec = train_and_evaluate(model_with_mec, train_images, train_labels, test_images, test_labels, use_augmentation=True)
    print(f'Accuracy with MEC Adjustments: {accuracy_with_mec:.4f}')
