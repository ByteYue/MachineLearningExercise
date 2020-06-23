from sklearn import neighbors
import os
import os.path
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import time


def train(train_dir, n_neighbors):
    """
    param train_dir: 训练数据路径
    param n_neighbors: n的数目
    return: 训练所得分类器
    """
    X = []# face_encoding
    y = []# dir

    print("Start Training")
    start = time.time()
    # 以每一个人名文件夹为一次循环开始遍历
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 遍历当前文件夹的人像
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image, model="cnn")

            if len(face_bounding_boxes) != 1:
                # 一张照片的人像不为1
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    Kclassifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="kd_tree", weights='distance', p =4)
    Kclassifier.fit(X, y)

    print("Cost:" + str(time.time()-start))
    return Kclassifier

def testData(test_dir):
    """
    param test_dir: 测试数据路径
    return: 
    """
    X = []# face_encoding
    Y = []# dir
    Z= []# face_location
    # 以每一个人名文件夹为一次循环开始遍历
    for class_dir in os.listdir(test_dir):
        if not os.path.isdir(os.path.join(test_dir, class_dir)):
            continue

        # 遍历当前文件夹的人像
        for img_path in image_files_in_folder(os.path.join(test_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # 一张照片的人像不为1
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes))
                Y.append(class_dir)
                Z.append(face_bounding_boxes)

    return X,Y,Z

def predict(X_frames, X_labels,X_locations,result ,Kclassifier=None, distance_threshold=0.4):
    """
    :param X_frames: 预测画面encodings
    :param X_labels: 预测画面的正确标签
    :param X_locations:预测画面locations
    :param Kclassifier: 训练所得分类器
    :param result: 预测画面对应的结果array(0表示error,1表示success)
    :param distance_threshold: 容错率，越低越严格
    :return: [(name, bounding box),...]
    """
    if Kclassifier is None:
        raise Exception("Classifier needed")

    for i in range(len(X_frames)):
        closest_distances = Kclassifier.kneighbors(X_frames[i], n_neighbors=2)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_locations[i]))]
        if are_matches:
            if Kclassifier.predict(X_frames[i])==X_labels[i] :
                result[i]=1
    return np.sum(result)/result.size


if __name__ == "__main__":
    print("Training KNN classifier")
    classifier = train("train", n_neighbors=2)
    print("Training complete!")
    X_frames, X_labels , X_locations= testData("test")
    result=np.zeros(len(X_frames), dtype=np.int)
    rate = predict(X_frames, X_labels, X_locations, result, classifier, distance_threshold=0.4)
    print(rate)
