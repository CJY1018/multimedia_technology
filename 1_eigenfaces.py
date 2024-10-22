import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image


# 加载 ORL 数据集
def load_orl_faces(data_dir):
    images = []
    labels = []
    for subject_dir in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_dir)
        if os.path.isdir(subject_path):
            for image_name in os.listdir(subject_path):
                image_path = os.path.join(subject_path, image_name)
                image = Image.open(image_path).convert('L')  # 转为灰度图
                image = np.array(image).flatten()  # 将图像展平
                images.append(image)
                labels.append(int(subject_dir[1:]))  # 从目录名中提取标签
    return np.array(images), np.array(labels)


# 可视化准确率曲线
def plot_accuracy_curve(accuracy_scores):
    plt.plot(range(1, n_components + 1), accuracy_scores, '-o')
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.show()


# 可视化特征脸
def plot_eigenfaces(eigenfaces, h, w, n_row=4, n_col=10):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    for i in range(n_row * n_col):
        if i < len(eigenfaces):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
    plt.suptitle("Eigenfaces", size=16)
    plt.show()


# 可视化重构图像
def plot_reconstructed_images(restore_images, accuracy_scores):
    plt.figure(figsize=(10, 12))
    plt.suptitle('Reconstructed Images with different n_components', size=18)
    for i, image in enumerate(restore_images):
        plt.subplot(n_components // 5 + n_components % 5, 5, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'n_components: {i + 1}')
        plt.xticks(())
        plt.yticks(())
        plt.xlabel(f'accuracy: {accuracy_scores[i] * 100:.2f}%', color='red')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    data_dir = './archive'  # ORL 数据集的目录
    images, labels = load_orl_faces(data_dir)

    # 图像尺寸
    h, w = 112, 92  # ORL 数据集中每张图片的尺寸
    n_components = 40  # 保留的主成分数量

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

    accuracy_scores = []
    eigenfaces = []
    restore_images = []

    # 逐个增加主成分数量
    for n_component in range(1, n_components + 1):
        # 使用PCA降维
        pca = PCA(n_components=n_component, svd_solver='auto', whiten=True).fit(X_train)

        # 将训练集和测试集投影到主成分空间
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # 使用支持向量机进行分类
        svm_clf = SVC(kernel='linear', class_weight='balanced')
        svm_clf.fit(X_train_pca, y_train)

        # 预测测试集并计算准确度
        y_pred = svm_clf.predict(X_test_pca)
        # print(f'ground_truth: {y_test},\n\n result:{y_pred}')
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
        eigenfaces.append(pca.components_)
        # 读取第一张图像并重构
        restore_images.append(pca.inverse_transform(pca.transform([images[0]])).reshape(h, w))
        print(f"n_components: {n_component}, accuracy: {accuracy * 100:.2f}%")

    # 可视化准确率曲线
    plot_accuracy_curve(accuracy_scores)

    # 可视化特征脸
    n_component = 20  # 设置需要可视化的特征脸数量
    plot_eigenfaces(eigenfaces[n_component - 1], h, w, n_row=(n_component - 1) // 10 + 1, n_col=10)

    # 可视化重构图像
    plot_reconstructed_images(restore_images, accuracy_scores)
