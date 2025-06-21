import os
import cv2
import numpy as np

vertical_filter = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
horizontal_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def edge_detection(img):
    if img is None:
        raise ValueError("Image is None, cannot perform edge detection.")
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)  # 将其转换为三通道图像
    n, m, d = img.shape
    edges_img = img.copy()
    for row in range(3, n-2):
        for col in range(3, m-2):
            local_pixels = img[row-1:row+2, col-1:col+2, 0]
            vertical_transformed_pixels = vertical_filter * local_pixels
            vertical_score = vertical_transformed_pixels.sum() / 4
            horizontal_transformed_pixels = horizontal_filter * local_pixels
            horizontal_score = horizontal_transformed_pixels.sum() / 4
            edge_score = (vertical_score**2 + (1.5 * horizontal_score)**2)**.5
            edges_img[row, col] = [edge_score] * 3
    edges_img = edges_img / edges_img.max() * 255  # 归一化并转换为0-255范围
    return edges_img.astype(np.uint8)

def save_edge_image(img, filename):
    cv2.imwrite(filename, img)

def create_dataset_splits(root_dir, train_dir, test_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    class_counter = 1
    for person_id in range(1, 157):
        for finger_id in ['f1', 'f2']:
            person_finger_dir = os.path.join(root_dir, str(person_id), finger_id)
            if not os.path.exists(person_finger_dir):
                continue
            # Create directories for this class
            train_class_dir = os.path.join(train_dir, f'{class_counter:03d}')
            test_class_dir = os.path.join(test_dir, f'{class_counter:03d}')
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)
            # Check for first collection
            first_collection_dir = os.path.join(person_finger_dir, '1')
            if os.path.exists(first_collection_dir):
                images = sorted(os.listdir(first_collection_dir))
                for i, img in enumerate(images):
                    src = os.path.join(first_collection_dir, img)
                    try:
                        # 读取图片
                        img_data = cv2.imread(src)
                        if img_data is None:
                            print(f"Error reading image: {src}")
                            continue
                        # 进行边缘检测
                        edges_img = edge_detection(img_data)
                        # 保存处理后的图片
                        if i < 4:
                            dst = os.path.join(train_class_dir, f'{img[:-4]}.bmp')
                        else:
                            dst = os.path.join(test_class_dir, f'{img[:-4]}.bmp')
                        save_edge_image(edges_img, dst)
                    except FileNotFoundError as e:
                        print(e)
            # Check for second collection
            second_collection_dir = os.path.join(person_finger_dir, '2')
            if os.path.exists(second_collection_dir):
                images = sorted(os.listdir(second_collection_dir))
                for i, img in enumerate(images):
                    src = os.path.join(second_collection_dir, img)
                    try:
                        # 读取图片
                        img_data = cv2.imread(src)
                        if img_data is None:
                            print(f"Error reading image: {src}")
                            continue
                        # 进行边缘检测
                        edges_img = edge_detection(img_data)
                        # 保存处理后的图片
                        if i < 4:
                            dst = os.path.join(train_class_dir, f'{img[:-4]}.bmp')
                        else:
                            dst = os.path.join(test_class_dir, f'{img[:-4]}.bmp')
                        save_edge_image(edges_img, dst)
                    except FileNotFoundError as e:
                        print(e)
            print("已完成{}".format(class_counter))
            class_counter += 1

if __name__ == "__main__":
    root_dir = '../HKPU/Original'
    train_dir = '../fingerData/hkpu_data_divide_edge/train'
    test_dir = '../fingerData/hkpu_data_divide_edge/test'
    create_dataset_splits(root_dir, train_dir, test_dir)
