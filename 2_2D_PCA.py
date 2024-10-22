import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


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
                images.append(image)
                labels.append(int(subject_dir[1:]))  # 从目录名中提取标签
    return np.array(images), np.array(labels)


# 2D PCA
def pca_2d(images, n_components):
    # 计算均值图像
    mean_image = np.mean(images, axis=0)
    images = images - mean_image

    # 计算协方差矩阵
    sum_cov = np.zeros((images.shape[2], images.shape[2]))
    for i in range(images.shape[0]):
        sum_cov += np.dot(images[i].T, images[i])

    sum_cov /= images.shape[0]

    # 计算特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(sum_cov)

    # 按特征值降序排列，选择前 n_components 个特征向量
    idx = np.argsort(eig_values)[::-1]
    eig_vectors = eig_vectors[:, idx]
    eig_vectors = eig_vectors[:, :n_components]

    return eig_vectors, mean_image


# 特征提取和图像重构
def extract_and_reconstruct(image, eig_vectors):
    # 提取特征
    features = np.dot(image, eig_vectors)
    features_flatten = features.flatten()

    # 重构图像
    reconstructed_image = np.dot(features, eig_vectors.T)
    return features_flatten, reconstructed_image


if __name__ == '__main__':
    data_dir = './archive'  # ORL 数据集的目录
    images, labels = load_orl_faces(data_dir)

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=7859)

    reconstructed_images = []
    accuracies = []
    for n_components in range(1, 21):
        # 使用 2D PCA 进行降维
        eig_vectors, mean_image = pca_2d(images, n_components)

        # 对训练集和测试集进行特征提取
        train_features = np.array([extract_and_reconstruct(img, eig_vectors)[0] for img in X_train])
        test_features = np.array([extract_and_reconstruct(img, eig_vectors)[0] for img in X_test])

        # 使用支持向量机进行分类
        svm_clf = SVC(kernel='linear', class_weight='balanced')
        svm_clf.fit(train_features, y_train)
        accuracy = svm_clf.score(test_features, y_test)
        accuracies.append(accuracy)
        print(f'n_components: {n_components}, accuracy: {accuracy * 100:.2f}%')

        # 提取特征并重构图像
        _, reconstructed_image = extract_and_reconstruct(images[0], eig_vectors)
        reconstructed_images.append(reconstructed_image)

    # 可视化重构图像，图像底部标注准确率
    plt.figure(figsize=(10, 12))
    plt.suptitle('Reconstructed Images with different n_components', size=18)
    for i, image in enumerate(reconstructed_images):
        plt.subplot(len(accuracies) // 5 + len(accuracies) % 5, 5, i + 1)
        plt.imshow(image.reshape(112, 92), cmap='gray')
        plt.title(f'n_components: {i + 1}')
        plt.xticks(())
        plt.yticks(())
        # 红色标注准确率
        plt.xlabel(f'accuracy: {accuracies[i] * 100:.2f}%', color='red')
    plt.tight_layout()
    plt.show()

    # 可视化准确率曲线
    plt.plot(range(len(accuracies)), accuracies, '-o')
    plt.xlabel('n_components')
    plt.ylabel('accuracy')
    plt.title('Accuracy with different n_components')
    plt.show()
