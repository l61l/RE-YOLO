from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import cv2
import numpy as np

def adjust_brightness(image_path, brightness):
    """
    调整图像亮度
    :param image_path: 图像路径
    :param brightness: 亮度调整值，范围-100到100
    :return: 调整亮度后的图像
    """
    image = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(image)
    brighter_image = enhancer.enhance(brightness / 100.0)
    return brighter_image

def add_noise(image_path):
    """
    给图像增加噪声
    :param image_path: 图像路径
    :return: 增加噪声后的图像
    """
    image = Image.open(image_path)
    noise_img = image.filter(ImageFilter.Noise(ImageFilter.Mode.RGB))
    return noise_img

def invert_image(image_path):
    """
    反转图像
    :param image_path: 图像路径
    :return: 反转后的图像
    """
    image = Image.open(image_path)
    inverted_image = ImageOps.mirror(image)
    return inverted_image

def rotate_image(image_path, angle):
    """
    旋转图像
    :param image_path: 图像路径
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    image = Image.open(image_path)
    rotated_image = image.rotate(angle)
    return rotated_image


def add_noise(image, noise_type='gaussian', strength=0.1):
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        var = 1
        sigma = strength ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_and_pepper':
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = strength
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out.astype(np.uint8)
    else:
        raise ValueError("Unsupported noise type")



def translate_image(image, shift):
    """
    平移图像
    
    参数：
    image: PIL Image对象，要平移的图像
    shift: 一个包含两个整数的元组，表示水平和垂直方向的平移量（x, y）
    
    返回值：
    平移后的图像
    """
    # 创建一个和原始图像大小相同的空白图像
    translated_image = Image.new("RGB", image.size)
    
    # 将原始图像粘贴到空白图像中，并进行平移
    translated_image.paste(image, shift)
    
    return translated_image

def crop_image(image, box):
    """
    裁剪图像
    
    参数：
    image: PIL Image对象，要裁剪的图像
    box: 一个包含四个整数的元组，表示裁剪框的位置和大小（left, upper, right, lower）
    
    返回值：
    裁剪后的图像
    """
    cropped_image = image.crop(box)
    return cropped_image


def apply_random_masks_inside(image, num_masks=5, mask_size=(80, 80)):
    """
    随机在图像内部生成多个遮挡
    
    参数：
    image: 输入图像
    num_masks: 要生成的遮挡数量，默认为 3
    mask_size: 遮挡的大小，默认为 (50, 50)
    
    返回值：
    带有遮挡的图像
    """
    # 获取图像大小
    height, width = image.shape[:2]
    
    # 复制输入图像，以免修改原始图像
    masked_image = image.copy()
    
    # 在图像内部随机生成多个遮挡
    for _ in range(num_masks):
        # 随机生成遮挡的位置
        x = np.random.randint(mask_size[0], width - mask_size[0])
        y = np.random.randint(mask_size[1], height - mask_size[1])
        
        # 创建遮挡
        mask = np.ones_like(image) * 255
        mask[y:y+mask_size[1], x:x+mask_size[0]] = 0
        
        # 将遮挡应用到图像上
        masked_image = cv2.bitwise_and(masked_image, mask)
    
    return masked_image







# 测试
image_path = "66.jpg"  # 替换为你的图像路径
brightness = 50  # 调整亮度值
noise_level = 2  # 噪声水平
angle = 25  # 旋转角度

# 调整亮度
# brighter_image = adjust_brightness(image_path, brightness)

# # 增加噪声
# noisy_image = add_noise(image_path)

# # 反转图像
# inverted_image = invert_image(image_path)

# # 旋转图像
# rotated_image = rotate_image(image_path, angle)

# 显示图像
# inverted_image.show()  # 显示旋转后的图像


# # 读取图像
# image = cv2.imread('66.jpg')

# # 添加噪声
# noisy_image = add_noise(image, noise_type='gaussian', strength=400)

# # 显示原始图像和添加噪声后的图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Noisy Image', noisy_image)
# # 保存图像
# cv2.imwrite('noisy_image.jpg', noisy_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 打开图像
# image = Image.open("66.jpg")

# # 平移图像
# shifted_image = translate_image(image, (50, 50))  # 水平和垂直方向都平移50个像素
# # 显示原始图像和平移后的图像
# shifted_image.show()
# # 保存平移后的图像
# shifted_image.save("shifted_image.jpg")
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 读取图像
image = cv2.imread('66.jpg')

# 应用随机内部遮挡
masked_image_inside = apply_random_masks_inside(image)

# 显示原始图像和带有内部遮挡的图像
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image Inside', masked_image_inside)
cv2.imwrite('masked_image_inside.jpg', masked_image_inside)
cv2.waitKey(0)
cv2.destroyAllWindows()