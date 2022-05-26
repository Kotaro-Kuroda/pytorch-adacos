import torch
import cv2
import pickle
from archs import archs
from torchvision import transforms
from torch.nn.functional import softmax
import config


def predict(model, image, height, width, classes):
    model.eval()
    model.cuda()
    img = cv2.imread(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229)),
    ])
    tensor_img = transform(img)
    tensor_img = tensor_img.cuda()
    tensor_img = tensor_img.unsqueeze(0)
    with torch.no_grad():
        output = model(tensor_img)
    probability = softmax(output[0], dim=0)
    pred = torch.argmax(probability)
    score = probability[pred]
    label = classes[pred]
    print(score.cpu().numpy(), label)


def main():
    model_path = '/home/kotarokuroda/Documents/pytorch-adacos/models/mnist_GhostNet_adacos_512d/model.pth'
    with open('/home/kotarokuroda/Documents/pytorch-adacos/models/mnist_GhostNet_adacos_512d/args.pkl', 'rb') as f:
        args = pickle.load(f)
    classes = config.classes
    model = archs.__dict__[args.arch](args, len(classes))
    model.load_state_dict(torch.load(model_path))
    image = '/home/kotarokuroda/Documents/pytorch-adacos/lion.jpg'
    predict(model, image, 512, 512, classes)


if __name__ == '__main__':
    main()
