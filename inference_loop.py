import torch
import torch.optim as optim
import warnings
# from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
from utils.util import *
from Encoder.e4e import load_e4e
from stylegan.generator import Generator
from CLIP.CLIP_Model import CLIP
from Segmentation_Model.segmentation_model import *
from densepose.model import DenseNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'

e4e_model = load_e4e('checkpoints/e4e.pt').to(device)
densepose_model = DenseNet('checkpoints/densepose.pkl').to(device)
Seg_Model = SegModel('checkpoints/segmentator.pt').to(device)
clip =CLIP().to(device)

checkpoint = torch.load('checkpoints/model.pt')

def train_styleclip(generator, text_prompt, num_epochs, learning_rate, device, image_path, clip_w, img_w, reg_w, body_shape_w,head_w,stitch):
    torch.autograd.set_detect_anomaly(True)
    generator.train(False)
    latent_code = make_latent_code(e4e_model,image_path).to(device)
    latent_code.requires_grad = True
    optimizer = optim.Adam([latent_code], lr=learning_rate)

    losses = []
    with torch.no_grad():
      text_feats = clip.encode_text(text_prompt).view(1, -1)
      shape_real = densenet_forward(densepose_model, for_model(image_path).to(device))
      real_bg, real_body, real_head = Seg_Model(for_model(image_path).to(device))
    blend_mask = 1 - real_head

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(num_epochs):
            warnings.filterwarnings("ignore", module='semantic_loss')
            warnings.filterwarnings("ignore", module='for_model')
            warnings.filterwarnings("ignore", message="Setting up PyTorch plugin 'upfirdn2d_plugin'... Failed!")
            optimizer.zero_grad()
            img = generator.synthesis(latent_code)
            if torch.isnan(img).any():
              print(f"NaN detected in generated image at epoch {epoch}")
              break
            body_shape = densenet_forward(densepose_model,img)
            body_shape_loss = F.mse_loss(shape_real, body_shape, reduction='none').mean((1,2,3)).sum()
            clip_sim = clip_similarity(clip,img, text_feats)
            clip_loss = (1 - clip_sim).sum()
            image_loss, head_loss = semantic_loss('checkpoints/segmentator.pt',img,image_path,device)
            reg_loss = w_regularization_loss(latent_code)
            net_loss = clip_w*clip_loss + head_loss*head_w + img_w*image_loss + reg_w*reg_loss + body_shape_loss*body_shape_w
            losses.append(net_loss.item())
            net_loss.backward(retain_graph=True)
            if torch.isnan(net_loss).any():
                print(f"NaN detected in net_loss at epoch {epoch}")
                break
            optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {net_loss.item()}")

    final_image = generator.synthesis(latent_code)
    # print(final_image.shape)
    if stitch:
        final_image = blend_mask*final_image + (1-blend_mask)*(for_model(image_path).to(device))  # Image Stitching

    final_name = f"Clip_W={clip_w},Img_W={img_w},Reg_W={reg_w},Stitch={stitch},Epochs={num_epochs},lr={learning_rate},{text_prompt}".replace('.',',')
    final_name += '.png'
    # save_image(final_image, final_name, normalize=True)
    print("Training complete")
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(final_image.view(3,256,256).permute(1,2,0).detach().cpu().numpy())
    plt.axis('off')
    plt.title(text_prompt)
    plt.tight_layout()
    plt.show()

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{num_epochs} Epochs')
    plt.show()


def main(num_epochs, lr, clip_w, img_w, reg_w,body_shape_w,head_w, stitch, image_path, text_prompt):
    generator = Generator(
        z_dim= 512,
        w_dim= 512,
        mapping_kwargs= {'num_layers': 2},
        synthesis_kwargs= {'channel_base': 16384,'channel_max': 512,'num_fp16_res': 4,'conv_clamp': 256},
        c_dim= 0,
        img_resolution= 256,
        img_channels= 3) # Adjust parameters as needed
    generator.load_state_dict(checkpoint['G'])
    generator = generator.to(device)
    train_styleclip(generator, text_prompt, num_epochs, lr, device,image_path, clip_w, img_w, reg_w,body_shape_w,head_w, stitch)