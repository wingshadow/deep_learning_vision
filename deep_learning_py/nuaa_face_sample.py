import datetime
import os

import cv2
import pandas as pd
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


def read_train_file():
    base_path = r'D:\ml\fas\data\NUAA'
    train_file_path = os.path.join(base_path, "train.txt")
    train_image_path_list = []
    train_labels_list = []
    with open(train_file_path) as f:
        lines = f.readlines()
        for line in lines:
            image_path = line.split(',')[0]
            label = line.split(',')[1]
            train_image_path_list.append(image_path)
            train_labels_list.append(label)
    train_df = pd.DataFrame(list(zip(train_image_path_list, train_labels_list)), columns=['image', 'label'])
    return train_df


def read_val_file():
    base_path = r'D:\ml\fas\data\NUAA'
    val_file_path = os.path.join(base_path, "val.txt")
    val_image_path_list = []
    val_labels_list = []
    with open(val_file_path) as f:
        lines = f.readlines()
        for line in lines:
            image_path = line.split(',')[0]
            label = line.split(',')[1]
            val_image_path_list.append(image_path)
            val_labels_list.append(label)
    val_df = pd.DataFrame(list(zip(val_image_path_list, val_labels_list)), columns=['image', 'label'])
    return val_df


def read_test_file():
    base_path = r'D:\ml\fas\data\NUAA'
    test_file_path = os.path.join(base_path, "test.txt")
    test_image_path_list = []
    test_labels_list = []
    with open(test_file_path) as f:
        lines = f.readlines()
        for line in lines:
            image_path = line.split(',')[0]
            label = line.split(',')[1]
            test_image_path_list.append(image_path)
            test_labels_list.append(label)
    test_df = pd.DataFrame(list(zip(test_image_path_list, test_labels_list)), columns=['image', 'label'])
    return test_df


def data_processing(train_df, val_df, test_def):
    # 将所有图像乘以 1/255 缩放
    # rotation_range 是角度值（在 0~180 范围内），表示图像随机旋转的角度范围
    # width_shift 和 height_shift 是图像在水平或垂直方向上平移的范围（相对于总宽度或总高度的比例）
    # shear_range 是随机错切变换的角度
    # zoom_range 是图像随机缩放的范围
    # horizontal_flip 是随机将一半图像水平翻转
    # fill_mode是用于填充新创建像素的方法
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip='true')
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # 将所有图像的大小调整为 64×64
    x_train = train_datagen.flow_from_dataframe(dataframe=train_df, x_col='image', y_col='label',
                                                target_size=(64, 64))
    x_val = val_datagen.flow_from_dataframe(dataframe=val_df, x_col='image', y_col='label', target_size=(64, 64))
    x_test = val_datagen.flow_from_dataframe(dataframe=test_def, x_col='image', y_col='label', target_size=(64, 64))
    return x_train, x_val, x_test


def model1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer=Adam(1e-3), loss=categorical_crossentropy, metrics=['acc'])
    return model


def get_test_data():
    base_path = r'E:\ml\fas\data\NUAA'
    file_path = os.path.join(base_path, "test.txt")
    img_list = []
    labels_list = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            image_path = line.split(',')[0]
            label = line.split(',')[1]
            img = cv2.imread(image_path)
            img_list.append(img)
            labels_list.append(label)
    return img_list, labels_list


if __name__ == '__main__':
    # img_list, labels_list = get_test_data()
    # print(len(img_list), len(labels_list))
    train_df = read_train_file()
    val_df = read_val_file()
    test_df = read_test_file()
    x_train, x_val, x_test = data_processing(train_df, val_df, test_df)
    log_dir = "logs/face_dect/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 创建tensorboard callback 回调
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = model1()
    model.fit_generator(x_train, validation_data=x_val, epochs=20, verbose=2, callbacks=[tensorboard_callback])
    test_accuracy = model.evaluate_generator(x_test)
    print('Test accuracy is : ', test_accuracy, '%')
