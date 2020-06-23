# ML exercise records

## KNN实验
起初我自己借助网上的资源写了一个[myKNN](./KNN/myKNN.py)的文件，经过测试我的代码的效果并不是很理想，6000 1000的情况下会跑接近一分钟， 60000 10000的情况下则是会跑80分钟左右。
之后我翻阅了不少相关资料后得知解决此类问题的相关数据结构KD-tree，同时也得知了sklearn库中有，便使用了此库进行了另一次编码[KNN](./KNN/myKNN.py)效果比较理想。

## 人脸识别实验
人脸识别部分分别实现了[KNN实现](/FaceRecognition/facereconition_knn.py)以及[直接比较人脸特征](/FaceRecognition/facerecognition.py),同时为了测试正确率,实现了[直接读取test文件](/FaceRecognition/ultimate.py)