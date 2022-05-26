from archs import archs
import pickle
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
import torchvision.transforms as transforms
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import cv2
import config
from torchvision.utils import make_grid, save_image

# Grad-CAM


def visualize(image, model, height, width, classes):
    model.eval()
    model.cuda()
    images = []
    target_layer = model.features
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMpp(model, target_layer)
    img = cv2.imread(image, 0)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width)),
    ])
    tensor_img = transform(img)
    normed_torch_img = transforms.Normalize((0.5), (0.25))(tensor_img)[None]
    tensor_img = tensor_img.cuda()
    normed_torch_img = normed_torch_img.cuda()
    tensor_img = tensor_img.unsqueeze(0)
    normed_torch_img = normed_torch_img
    mask, _ = gradcam(normed_torch_img)
    heatmap, result = visualize_cam(mask, tensor_img)
    mask_pp, _ = gradcam_pp(normed_torch_img)
    heatmap_pp, result_pp = visualize_cam(mask_pp, tensor_img)
    images.extend([heatmap, heatmap_pp, result, result_pp])
    grid_image = make_grid(images, nrow=5)
    image = transforms.ToPILImage()(grid_image)
    image.save('./res.jpg')
    with torch.no_grad():
        output = model(tensor_img)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred = torch.argmax(probs)
    print(probs[0][pred], classes[pred])


def main():
    model_path = '/home/kotarokuroda/Documents/pytorch-adacos/models/mnist_ResNet_adacos_512d/model.pth'
    with open('/home/kotarokuroda/Documents/pytorch-adacos/models/mnist_ResNet_adacos_512d/args.pkl', 'rb') as f:
        args = pickle.load(f)
    classes = config.classes
    model = archs.__dict__[args.arch](args, len(classes))
    model.load_state_dict(torch.load(model_path))
    image = '/home/kotarokuroda/Documents/xray_dataset_covid19/test/PNEUMONIA/streptococcus-pneumoniae-pneumonia-temporal-evolution-1-day1.jpg'
    visualize(image, model, 114, 114, classes)


if __name__ == '__main__':
    main()
