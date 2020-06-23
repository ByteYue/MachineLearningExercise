import cv2
from sklearn import neighbors
import os
import os.path
from PIL import Image, ImageDraw
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
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # 一张照片的人像不为1
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    Kclassifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm="kd_tree", weights='distance', p =2)
    Kclassifier.fit(X, y)

    print("Cost:" + str(time.time()-start))
    return Kclassifier


def predict(X_frame, Kclassifier=None, distance_threshold=0.4):
    """
    :param X_frame: 预测画面
    :param Kclassifier: 训练所得分类器
    :param distance_threshold: 容错率，越低越严格
    :return: [(name, bounding box),...]
    """
    if Kclassifier is None:
        raise Exception("Classifier needed")

    X_face_locations = face_recognition.face_locations(X_frame)

    # 如果没发现人脸则返回空
    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    closest_distances = Kclassifier.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(Kclassifier.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    """
    :param frame: 显示画面
    :param predictions: 预测结果
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


if __name__ == "__main__":
    print("Training KNN classifier")
    classifier = train("train", n_neighbors=2)
    print("Training complete!")
    flag = 19 #每20张画面处理一张
    print('loading camera')
    cap = cv2.VideoCapture(-1)
    while True:
        val, frame = cap.read()
        if val:
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            flag=flag+1
            if flag % 20 == 0:
                predictions = predict(img, Kclassifier= classifier)
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
