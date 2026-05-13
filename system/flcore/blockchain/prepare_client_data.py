"""
准备客户端数据的脚本
修复图像读取错误，并将数据分配给10个客户端
"""
import os
import cv2
import numpy as np

# 设置客户端数量
client_num = 10
client_folder = 'Client_datasets_10'

# 定义安全的图像读取函数
def safe_load_image(image_path):
    """
    安全地读取图像文件，处理读取失败的情况
    :param image_path: 图像文件路径
    :return: 处理后的图像数组或None
    """
    try:
        img = cv2.imread(image_path, 0)  # 以灰度模式读取
        if img is None or img.size == 0:
            return None
        img = cv2.resize(img, (28, 28))  # resize到MNIST标准尺寸
        img = img.astype(np.float32) / 255  # 归一化
        return img
    except Exception as e:
        print(f"警告: 读取图像失败 {image_path}: {e}")
        return None

# 创建客户端文件夹
print(f"创建 {client_num} 个客户端文件夹...")
for i in range(1, client_num + 1):
    os.makedirs(os.path.join(client_folder, f'client_{i}'), exist_ok=True)
print(f"[OK] 已创建 {client_num} 个客户端文件夹\n")

# 获取每个类别数据的图像数据（带错误处理）
print("开始读取图像数据...")
number_list = []

for digit in range(10):
    digit_folder = f"mnist_train/{digit}"
    if not os.path.exists(digit_folder):
        print(f"错误: 文件夹 {digit_folder} 不存在！")
        continue
        
    images = []
    file_list = os.listdir(digit_folder)
    # 只处理PNG文件
    png_files = [f for f in file_list if f.lower().endswith('.png')]
    
    print(f"处理数字 {digit}，共 {len(png_files)} 个PNG文件...", end=' ')
    failed_count = 0
    
    for filename in png_files:
        image_path = os.path.join(digit_folder, filename)
        img = safe_load_image(image_path)
        if img is not None:
            images.append([img])
        else:
            failed_count += 1
    
    number_list.append(images)
    print(f"成功: {len(images)}, 失败: {failed_count}")

# 解包到单独的变量（保持兼容性）
number_0, number_1, number_2, number_3, number_4, number_5, number_6, number_7, number_8, number_9 = number_list

# 每个类别的样本总数除以客户端数量
first_round_number = [len(number_list[i]) // client_num for i in range(10)]

# 每个类别剩余的样本数量
remain_number = [len(number_list[i]) % client_num for i in range(10)]

# 处理剩余数据
remain_data = []
remain_label = []
for digit in range(10):
    remain_count = remain_number[digit]
    if remain_count > 0:
        remain_data.extend(number_list[digit][-remain_count:])
        remain_label.extend([digit] * remain_count)

print(f"\n数据统计:")
print(f"客户端数量: {client_num}")
for i in range(10):
    print(f"数字 {i}: 总数={len(number_list[i])}, 每客户端={first_round_number[i]}, 剩余={remain_number[i]}")
print(f"剩余数据总数: {len(remain_data)}\n")

# 开始构建10个客户端的数据
print("开始分配数据到客户端...")
for i in range(client_num):
    data = []
    label = []
    
    # 为每个类别分配数据
    for digit in range(10):
        start_idx = i * first_round_number[digit]
        end_idx = (i + 1) * first_round_number[digit]
        data.extend(number_list[digit][start_idx:end_idx])
        label.extend([digit] * first_round_number[digit])
    
    # 将剩余的数据再补充分配到每个数据集当中
    if len(remain_data) > 0:
        remain_start = i * len(remain_data) // client_num
        remain_end = (i + 1) * len(remain_data) // client_num
        data.extend(remain_data[remain_start:remain_end])
        label.extend(remain_label[remain_start:remain_end])
    
    # 保存每个客户端的数据
    # client_path = os.path.join(client_folder, f'client_{i+1}')
    
    # 标准化路径：保存到 ../../data/mnist/client_{i+1}
    # 假设脚本在 system/flcore/blockchain/ 运行
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'mnist')
    client_path = os.path.join(base_data_dir, f'client_{i+1}')
    
    os.makedirs(client_path, exist_ok=True)
    np.save(os.path.join(client_path, 'data.npy'), data, allow_pickle=True)
    np.save(os.path.join(client_path, 'label.npy'), label, allow_pickle=True)
    
    print(f"客户端 {i+1}: 数据形状={np.array(data).shape}, 标签形状={np.array(label).shape}")

print(f"\n[完成] 已为 {client_num} 个客户端准备好数据")
print(f"数据保存在: {client_folder}/")

