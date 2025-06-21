import os
import shutil

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
                    if i < 4:
                        dst = os.path.join(train_class_dir, f'{img}')
                    else:
                        dst = os.path.join(test_class_dir, f'{img}')
                    shutil.copy(src, dst)

            # Check for second collection
            second_collection_dir = os.path.join(person_finger_dir, '2')
            if os.path.exists(second_collection_dir):
                images = sorted(os.listdir(second_collection_dir))
                for i, img in enumerate(images):
                    src = os.path.join(second_collection_dir, img)
                    if i < 4:
                        dst = os.path.join(train_class_dir, f'{img}')
                    else:
                        dst = os.path.join(test_class_dir, f'{img}')
                    shutil.copy(src, dst)

            class_counter += 1

if __name__ == "__main__":
    root_dir = '../HKPU/Original'
    train_dir = '../fingerData/hkpu_data_divide/train'
    test_dir = '../fingerData/hkpu_data_divide/test'
    create_dataset_splits(root_dir, train_dir, test_dir)
