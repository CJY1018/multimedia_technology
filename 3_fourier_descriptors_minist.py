import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
import torch
from torch import nn, optim


# 读取minist数据集
def load_mnist(data_dir):
    mnist = datasets.MNIST(data_dir, train=True, download=True)
    images = mnist.data.numpy()
    # 二值化
    _, images = cv2.threshold(images, 128, 255, cv2.THRESH_BINARY)
    labels = mnist.targets.numpy()
    return images, labels


# 寻找形状的闭合边界，返回边界的坐标
def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# 计算傅里叶描述子
def fourier_descriptors(contour, num_descriptors):
    # 计算复数形式的傅里叶描述子
    descriptors = np.fft.fft(contour[:, 0, 0] + 1j * contour[:, 0, 1])

    # 保留前 num_descriptors 个描述子
    descriptors = descriptors[:num_descriptors]

    return descriptors


def create_feature_vector(descriptors, max_descriptors):
    feature_vector = []
    for desc in descriptors:
        # 保留前 max_descriptors 个描述子
        if len(desc) < max_descriptors:
            desc = np.pad(desc, (0, max_descriptors - len(desc)), 'constant')
        else:
            desc = desc[:max_descriptors]

        real_part = np.real(desc)
        imag_part = np.imag(desc)
        feature_vector.append(np.concatenate((real_part, imag_part)))

    return np.array(feature_vector)


# 分类
def classify(features, labels):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练分类器
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 输出分类报告
    print(classification_report(y_test, y_pred))


class FourierNN(nn.Module):
    def __init__(self, input_size):
        super(FourierNN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.seq(x)


# 使用神经网络进行分类
def classify_with_nn(features, labels, num_descriptors):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=7859)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 构建数据集加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 定义模型
    model = FourierNN(input_size=num_descriptors * 2)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    data_dir = './data'  # MNIST 数据集的目录
    images, labels = load_mnist(data_dir)
    num_descriptors = 60

    # 计算傅里叶描述子
    descriptors = []
    for image in images:
        # 寻找形状的边界
        contours = find_contours(image)
        if len(contours) > 0:  # 确保有找到的轮廓
            contour = contours[0]
            # 计算傅里叶描述子
            descriptors.append(fourier_descriptors(contour, num_descriptors=num_descriptors))

    # 创建特征向量
    features = create_feature_vector(descriptors, max_descriptors=num_descriptors)

    # 分类
    # classify(features, labels)

    # 训练神经网络.并分类
    classify_with_nn(features, labels, num_descriptors)

    # 重建图像
    reconstructed_images = []
    for desc in descriptors:
        contour = np.fft.ifft(desc)
        contour = np.array([contour.real, contour.imag]).T
        reconstructed_image = np.zeros((28, 28), dtype=np.uint8)
        cv2.drawContours(reconstructed_image, [contour.astype(np.int32)], -1, 255, thickness=-1)
        reconstructed_images.append(reconstructed_image)

    # 可视化
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(reconstructed_images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.xticks(())
        plt.yticks(())
    plt.show()
