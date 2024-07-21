import cv2
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from densepose.model import set_instances
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_latent_code(model,image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    tensor_image = transform(image).to(device)
    out = model(tensor_image.view(1,3,256,256))
    return out

def w_regularization_loss(latent_code):
    N = latent_code.shape[1]
    initial_latent_vector = latent_code[:,0,:].view(-1,1,512)
    w_ref = initial_latent_vector.repeat(1,N-1,1)
    w_tar = latent_code[:,1:,:]
    loss = F.mse_loss(w_ref, w_tar, reduction='none')
    loss = loss.mean((1, 2)).sum()
    return loss

def clip_similarity(model, imgs, text_feats):
    img_features = model.encode_image(imgs)
    clip_similarity = model.compute_similarity(img_features, text_feats)
    return clip_similarity

def densenet_forward(model, imgs):
    bs = imgs.size(0)
    box = torch.tensor([[0, 0, 256, 256]], device=imgs.device)
    instances = set_instances(box, bs)
    body = model(imgs, instances)
    return body