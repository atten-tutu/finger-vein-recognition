import os
from PIL import Image
import matplotlib.pyplot as plt

def edge(img, filename_pp, fcp):
    edges_img = img.copy()
    top1, top2, top3 = 0, 0, 0
    bot1, bot2, bot3 = 240, 240, 240

    # Assuming that the image has 3 channels (e.g., RGB)
    # You can select one channel (e.g., edges_img[row][30][0] for the red channel)
    for row in range(120, 0, -1):
        if top1 == 0 and edges_img[row][30][0] > 200:
            top1 = row
        if top2 == 0 and edges_img[row][110][0] > 200:
            top2 = row
        if top3 == 0 and edges_img[row][180][0] > 200:
            top3 = row
        if top1 != 0 and top2 != 0 and top3 != 0:
            break

    for row in range(120, 240):
        if bot1 == 240 and edges_img[row][30][0] > 200:
            bot1 = row
        if bot2 == 240 and edges_img[row][110][0] > 200:
            bot2 = row
        if bot3 == 240 and edges_img[row][180][0] > 200:
            bot3 = row
        if bot1 != 240 and bot2 != 240 and bot3 != 240:
            break

    top = max(top1, top2, top3)
    bot = min(bot1, bot2, bot3)

    if top == 0:
        top = 60
        print("Error Top:" + fcp)
    if bot == 240:
        bot = 180
        print("Error Bot:" + fcp)

    imgg = Image.open(filename_pp)
    cropped = imgg.crop((0, top, 320, bot))
    cropped.save(fcp)


def create_dataset_splits_with_roi(root_dir, train_dir, test_dir, edge_dir):
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
                    edge_img_path = os.path.join(edge_dir, 'train' if i < 4 else 'test', f'{class_counter:03d}', img[:-3] + "bmp")
                    roi_img_path = os.path.join(train_class_dir if i < 4 else test_class_dir, f'{img[:-3]}_roi.bmp')

                    # Perform edge detection and ROI extraction
                    if os.path.exists(edge_img_path):
                        edge_img = plt.imread(edge_img_path)
                    else:
                        print(f"File not found: {edge_img_path}")

                    edge(edge_img, src, roi_img_path)

            # Check for second collection
            second_collection_dir = os.path.join(person_finger_dir, '2')
            if os.path.exists(second_collection_dir):
                images = sorted(os.listdir(second_collection_dir))
                for i, img in enumerate(images):
                    src = os.path.join(second_collection_dir, img)
                    edge_img_path = os.path.join(edge_dir, 'train' if i < 4 else 'test', f'{class_counter:03d}', img[:-3] + "bmp")
                    roi_img_path = os.path.join(train_class_dir if i < 4 else test_class_dir, f'{img[:-3]}_roi.bmp')

                    # Perform edge detection and ROI extraction
                    if os.path.exists(edge_img_path):
                        edge_img = plt.imread(edge_img_path)
                    else:
                        print(f"File not found: {edge_img_path}")

                    edge(edge_img, src, roi_img_path)

            class_counter += 1

if __name__ == "__main__":
    root_dir = '../HKPU/Original'
    train_dir = '../fingerData/hkpu_data_divide_roi/train'
    test_dir = '../fingerData/hkpu_data_divide_roi/test'
    edge_dir = '../fingerData/hkpu_data_divide_edge'
    create_dataset_splits_with_roi(root_dir, train_dir, test_dir, edge_dir)
