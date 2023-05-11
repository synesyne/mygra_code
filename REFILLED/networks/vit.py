# 基本的ViT模型，并使用CIFAR-100数据集
# 使用depth个Transformer编码器层，最后接一个MLP预测类别。
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from einops import rearrange
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# in:8,3,32,32
# out:8,100
# 就是vit超参数的设置需要在这里改

# image_size = 32
# patch_size = 4
# num_classes = 100
# dim = 512
# depth = 6
# heads = 8
# mlp_dim = 1024
# pool = 'cls'
# channels = 3
# dim_head = 64
# dropout = 0.1
# emb_dropout = 0.1


class MyNetwork(nn.Module):
    def __init__(self, args):
    # def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_size = args.vit_image_size
        patch_size = args.vit_patch_size
        num_classes = args.vit_num_classes
        dim = args.vit_dim
        depth = args.vit_depth
        heads = args.vit_heads
        mlp_dim = args.vit_mlp_dim
        channels = args.vit_channels
        dim_head = args.vit_dim_head
        dropout = args.vit_dropout
        emb_dropout = args.vit_emb_dropout
        pool = 'cls'
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, flag_embedding=False, flag_both=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        # print(x.shape) # CUB-200是torch.Size([32, 3137, 512])  cifar100是torch.Size([32, 65, 512])
        # print((self.pos_embedding[:, :(n + 1)]).shape) # torch.Size([1, 65, 512])

        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        y = self.to_latent(x)

        if flag_embedding:
            return y
        else:
            l = self.mlp_head(y)
            if flag_both:
                return l, y
            else:
                return l
    def get_network_params(self):
        # modules = [self.to_patch_embedding, self.cls_token, self.pos_embedding, self.dropout, self.transformer, self.to_latent]
        modules = [self.to_patch_embedding, self.dropout, self.transformer, self.to_latent]
        # 这两个不行：self.cls_token, self.pos_embedding,
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j

    def get_classifier_params(self):
        modules = [self.mlp_head]
        for i in range(len(modules)):
            for j in modules[i].parameters():
                yield j

# if __name__ == "__main__":
#   # Set device
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   # print(str(device))

#   # Load CIFAR-100 dataset
#   transform = transforms.Compose(
#       [transforms.ToTensor(),
#       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#   # 由于GPU显存不足，将batchsize从64改为8
#   train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
#                                           download=False, transform=transform)
#   train_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
#                                             shuffle=True, num_workers=2)

#   test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                           download=False, transform=transform)
#   test_loader = torch.utils.data.DataLoader(test_set, batch_size=8,
#                                             shuffle=False, num_workers=2)

#   # Initialize ViT model
#   model = ViT(
#       image_size = 32,
#       patch_size = 4,
#       num_classes = 100,
#       dim = 512,
#       depth = 1, # 6,
#       heads = 1, # 8,
#       mlp_dim = 1024,
#       dropout = 0.1,
#       emb_dropout = 0.1
#   ).to(device)

#   # Define loss function and optimizer
#   criterion = nn.CrossEntropyLoss()
#   optimizer = optim.Adam(model.parameters(), lr=1e-3)


#   # Train the model
#   num_epochs = 8

#   for epoch in range(num_epochs):
#       train_loss = 0.0
#       train_acc = 0.0
#       model.train()
#       for batch in tqdm(train_loader):
#           images, labels = batch
#           images, labels = images.to(device), labels.to(device)
#         #   images.shape torch.Size([8, 3, 32, 32])
#         #   outputs.shape torch.Size([8, 100])

#           optimizer.zero_grad()
#           outputs = model(images)
#           loss = criterion(outputs, labels)
#           loss.backward()
#           optimizer.step()
#           train_loss += loss.item() * images.size(0)
#           _, preds = torch.max(outputs, 1)
#           train_acc += torch.sum(preds == labels.data)
#       train_loss /= len(train_loader.dataset)
#       train_acc /= len(train_loader.dataset)
      
#       test_loss = 0.0
#       test_acc = 0.0
#       model.eval()
#       with torch.no_grad():
#           for batch in tqdm(test_loader):
#               images, labels = batch
#               images, labels = images.to(device), labels.to(device)
#               outputs = model(images)
#               loss = criterion(outputs, labels)
#               test_loss += loss.item() * images.size(0)
#               _, preds = torch.max(outputs, 1)
#               test_acc += torch.sum(preds == labels.data)
#           test_loss /= len(test_loader.dataset)
#           test_acc /= len(test_loader.dataset)
      
#       print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

#   print('Finished Training')

# # Epoch 1/1, Train Loss: 4.4189, Train Acc: 0.0324, Test Loss: 4.4319, Test Acc: 0.0258
# # Finished Training