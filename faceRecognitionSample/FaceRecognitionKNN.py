"""
使用k-最近邻（KNN）算法进行人脸识别的示例
"""

from matplotlib import pyplot as plt
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    训练k近邻分类器进行人脸识别。
    :param train_dir: 包含每个已知人员及其人脸的目录。
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (可选) 模型保存目录
    :param n_neighbors: (可选) 分类中要加权的邻居数。如果未指定，则自动选择，就是k-NN的k的值，选取最近的k个点
    :param knn_algo: (可选) knn底层的搜索算法
    :param verbose: 打印训练信息
    :return: 返回训练好的模型
    """
    X = []
    y = []

    # 读取人员路径
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 读取当前人员的人脸图片
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            # 加载图片
            image = face_recognition.load_image_file(img_path)
            # 人脸检测
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # 没有人就跳过当前图片
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # 保存人脸特征和类别
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # 自定设置n_neighbors
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # 训练KNN分类器
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # 保存分类器
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    使用经过训练的KNN分类器识别给定图像中的人脸
    :param X_img_path: 输入图像
    :param knn_clf: (可选) knn模型，和model_path必须有一个可用
    :param model_path: (可选) knn模型路径，和knn_clf必须有一个可用
    :param distance_threshold: (可选) 人脸分类的距离阈值。阈值越大，就越容易误报。
    :return: 人脸对应的人名和其边界框
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # 加载模型
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 读取图片和进行人脸检测
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # 如果没有检测到人脸就返回空list
    if len(X_face_locations) == 0:
        return []

    # 提取人脸特征
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # 使用K近邻进行分类
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # 返回预测结果
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    预测结果可视化
    :param img_path: 预测图像
    :param predictions: 预测结果
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # 画框
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # 设置名字，需要用uft-8编码
        name = name.encode("UTF-8")

        # 标注人名
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw

    # jupyter 绘图
    # pil_image.show()
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # 训练图片下载地址：https://github.com/ageitgey/face_recognition/tree/master/examples/knn_examples
    # STEP 1 训练KNN分类器
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2 使用训练好的KNN分类器对测试的人脸图像进行识别
    for image_file in os.listdir("knn_examples/test"):
        # 待测试人脸图像路径
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # 用经过训练的分类器模型查找图像中的所有人
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # 打印结果
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # 展示结果
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
