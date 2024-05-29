import torch
from torch import nn
from .clip import clip
from .BLIP2_T5 import *
import torch.nn.functional as F


class ExpCLIP_Train(nn.Module):
    def __init__(self, args):
        super().__init__()

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        self.blip_instruction = args.instruction

        if args.load_model == 'CLIP_B32':
            self.clip_model, _ = clip.load("ViT-B/32", device)
        elif args.load_model == 'CLIP_B16':
            self.clip_model, _ = clip.load("ViT-B/16", device)
        elif args.load_model == 'CLIP_L14':
            self.clip_model, _ = clip.load("ViT-L/14", device)

        if args.load_model == 'CLIP_L14':
            self.projection_head = Linear_Matrix_L14()
        else:
            self.projection_head = Linear_Matrix()
 
        self.blip2 = get_blip2t5_model(device)
        self.tokens = get_tokens(self.blip_instruction, self.blip2.t5_tokenizer, args.batch_size, device)

    def forward(self, image):
        
        image_features = self.clip_model.encode_image(image)
        image_features = image_features.float()
        image_features = self.projection_head(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        _, text_features = self.blip2.generate({"image": image, "tokens": self.tokens})

        text_features = torch.mean(text_features, dim=1)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()

        return logit_scale, image_features, text_features


class ExpCLIP_Test(nn.Module):
    def __init__(self, args):
        super().__init__()

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
        if args.load_model == 'CLIP_B32':
            self.clip_model, _ = clip.load("ViT-B/32", device)
        elif args.load_model == 'CLIP_B16':
            self.clip_model, _ = clip.load("ViT-B/16", device)
        elif args.load_model == 'CLIP_L14':
            self.clip_model, _ = clip.load("ViT-L/14", device)

        if args.load_model == 'CLIP_L14':
            self.projection_head = Linear_Matrix_L14()
        else:
            self.projection_head = Linear_Matrix()

    def forward(self, image, text=None, mode_task=None):
        
        if mode_task=='Static_FER':
            image_features = self.clip_model.encode_image(image)
            image_features = image_features.float()
            image_features = self.projection_head(image_features)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        elif mode_task=='Dynamic_FER':
            n, t, c, h, w = image.shape
            image = image.contiguous().view(-1, c, h, w)
            image_features = self.clip_model.encode_image(image)
            image_features = image_features.float()
            image_features = self.projection_head(image_features)
            image_features = image_features.reshape(n, t, -1)
            image_features = torch.mean(image_features, dim=1)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_tokenized = clip.tokenize(text, context_length=77, truncate=True).to('cuda')
        text_features = self.clip_model.encode_text(text_tokenized)
        text_features = text_features.float()
        text_features = self.projection_head(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.clip_model.logit_scale.exp()

        return logit_scale, image_features, text_features


class Linear_Matrix(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(512, 4096, bias=False)
    def forward(self, x):
        return self.mlp(x)


class Linear_Matrix_L14(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(768, 4096, bias=False)
    def forward(self, x):
        return self.mlp(x)