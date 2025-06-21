from torchvision import datasets, transforms
from PIL import Image, ImageEnhance

def get_dataset(train_dir, test_dir):

    def sharpen_image(img):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(2.0)  # 2.0 表示增加锐化，数值越高锐化越明显

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(5)], 0.05),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
        # transforms.RandomGrayscale(p=0.1),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 高斯模糊
        # transforms.Lambda(sharpen_image),  # 添加锐化处理
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.Lambda(sharpen_image),  # 添加锐化处理
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 训练集
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

    # 测试集
    eval_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    print("Finish Creating Dataset")

    return train_dataset, eval_dataset
