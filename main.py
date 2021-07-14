import cv2
import numpy as np
import torch
import generator


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 模型参数存放路径
model_path = 'model/generator.mdl'
# 调用模型结构
model = generator.Generator(3, 3, 64)
# model = generator.RRDBNet(3, 3, 64, 23, mid=32)
# 加载参数
# torch.load(model_path)
model.load_state_dict(torch.load(model_path), strict=True)

# 运行模式，非训练模式
model.eval()
# 模型运行模式转换
model = model.to(device)


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    # dataLoader中设置的mean参数
    mean = [0.485,0.456,0.406]
    # dataLoader中设置的std参数
    std = [0.229,0.224,0.225]

    # dataLoader中设置的mean参数
    # mean = [0,0,0]
    # dataLoader中设置的std参数
    # std = [1,1,1]

    if not isinstance(input_image, np.ndarray):
        # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): #反标准化
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)



def re_batch(img):
    # 最大运行尺寸设定
    h_step = 600
    w_step = 600

    if img.shape[0] % h_step == 0:
        steps_of_h = int(img.shape[0] / h_step)
    else:
        steps_of_h = int(img.shape[0] / h_step) + 1

    if img.shape[1] % w_step == 0:
        steps_of_w = int(img.shape[1] / w_step)
    else:
        steps_of_w = int(img.shape[1] / w_step) + 1

    batch = []
    for h in range(steps_of_h):
        if h == steps_of_h - 1:
            for w in range(steps_of_w):
                if w == steps_of_w - 1:
                    batch.append(img[h * h_step:, w * w_step:, :])
                else:
                    batch.append(img[h * h_step:, w * w_step:(w + 1) * w_step, :])
        else:
            for w in range(steps_of_w):
                if w == steps_of_w - 1:
                    batch.append(img[h * h_step:(h + 1) * h_step, w * w_step:, :])
                else:
                    batch.append(img[h * h_step:(h + 1) * h_step, w * w_step:(w + 1) * w_step, :])

    return batch, (steps_of_h, steps_of_w)



def img_remake(res_batch, shape):
    h_first = True
    clip = 0
    for h in range(shape[0]):
        if h_first:
            w_first = True
            for w in range(shape[1]):
                if w_first:
                    w_img = res_batch[clip]
                    clip += 1
                    w_first = False
                else:
                    w_img = np.concatenate([w_img, res_batch[clip]], 1)
                    clip += 1
            h_img = w_img
            h_first = False
        else:
            w_first = True
            for w in range(shape[1]):
                if w_first:
                    w_img = res_batch[clip]
                    clip += 1
                    w_first = False
                else:
                    w_img = np.concatenate([w_img, res_batch[clip]], 1)
                    clip += 1
            h_img = np.concatenate([h_img, w_img], 0)


    return h_img





def process(img):
    # 图片预处理，在numpy格式和tensor格式间转换
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    return output



def run(img_path, save_path):
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # 切分图片生成batch_list返回
    img_batch, img_shape = re_batch(img)
    # 用于保存运行结果
    res_batch = []
    i = 0
    all_number = len(img_batch)
    for clip in img_batch:
        print("正在运行：{}%".format(i / all_number * 100))
        i += 1
        res = process(clip)
        res_batch.append(res)

    # 按序拼接
    res = img_remake(res_batch, img_shape)
    # 图像保存
    cv2.imwrite(save_path, res)







if __name__ == "__main__":
    # 读取图像路径
    path = r"test1.jpg"
    # 保存图像路径
    save_path = r"test1.jpg"
    run(path, save_path)

