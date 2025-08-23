import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch.optim as optim



torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=3,stride=1,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel,stride,padding)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel,stride,padding)
        self.relu = nn.ReLU(inplace = True)


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        skip_connection = self.relu(x)
        return skip_connection

class ConvBlockDec(nn.Module):
    def __init__(self,in_channels,out_channels,kernel=3,stride=1,padding=0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2, output_padding=0)
        self.conv1 = nn.Conv2d(out_channels * 2,out_channels,kernel,stride,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel,stride,padding=1)
        self.relu = nn.ReLU(inplace = True)
 
    def forward(self,x,skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class Unet(nn.Module):
    def __init__(self,in_channels=1,out_channels=2):
        super().__init__()
        self.enc1 = ConvBlock(in_channels,64)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.enc2 = ConvBlock(64,128)
        self.enc3 = ConvBlock(128,256)
        self.enc4 = ConvBlock(256,512)

        self.base = ConvBlock(512,1024)

        self.dec4 = ConvBlockDec(1024,512)
        self.dec3 = ConvBlockDec(512,256)
        self.dec2 = ConvBlockDec(256,128)
        self.dec1 = ConvBlockDec(128,64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self,x):
        s1 = self.enc1(x)
        x = self.maxpool(s1)
        s2 = self.enc2(x)
        x = self.maxpool(s2)
        s3 = self.enc3(x)
        x = self.maxpool(s3)
        s4 = self.enc4(x)
        x = self.maxpool(s4)

        x = self.base(x)

        x = self.dec4(x,s4)
        x = self.dec3(x,s3)
        x = self.dec2(x,s2)
        x = self.dec1(x,s1)

        x = self.final_conv(x)

        return x
    
  
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet().to(device)
 
class PerImageZScore(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        img = img.astype(np.float32)
        m = img.mean()
        s = img.std()
        if s < 1e-6:
            s = 1e-6
        return (img - m) / s

train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.ElasticTransform(p=0.5, alpha=200, sigma=20),
    PerImageZScore(),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.Resize(512, 512),
    PerImageZScore(),
    ToTensorV2(),
])

class CustomDataset(Dataset):
  def __init__(self,image_paths,mask_paths,transform = None):
      self.image_paths = image_paths
      self.mask_paths = mask_paths
      self.transform = transform

  def __len__(self):
      return len(self.image_paths)

  def __getitem__(self, idx):
      img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
      mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
      img = np.expand_dims(img, axis=-1)
      mask = np.expand_dims(mask, axis=-1)
      mask = (mask > 127).astype(np.int64)

      if self.transform:
          augmented = self.transform(image=img, mask=mask)
          img = augmented["image"]
          mask = augmented["mask"]

      return img.float(), mask.long()

all_images = sorted(glob("./data/images/*.jpg"))
all_masks = sorted(glob("./data/labels/*.jpg"))

train_size = int(0.8 * len(all_images))
test_size = len(all_images) - train_size

train_images, test_images = all_images[:train_size], all_images[train_size:]
train_masks, test_masks = all_masks[:train_size], all_masks[train_size:]

train_dataset = CustomDataset(train_images, train_masks, transform=train_transform)
test_dataset = CustomDataset(test_images, test_masks, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,pin_memory=True)

optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
 

def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader,desc="Training", leave=False)
    total_loss = 0
    for imgs, masks in loop:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        masks = masks.squeeze(-1).long().to(device)
        loss = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def eval_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_pixels = 0
    correct_pixels = 0
    dice_score = 0
    iou_score = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            masks = masks.squeeze(-1).long().to(device)
            total_loss += loss_fn(preds, masks).item()

            preds_classes = torch.argmax(preds, dim=1)   
            correct_pixels += (preds_classes == masks).sum().item()
            total_pixels += torch.numel(preds_classes)
            intersection = (preds_classes & masks).float().sum((1, 2))
            union = (preds_classes | masks).float().sum((1, 2))
            dice = (2. * intersection + 1e-7) / (preds_classes.float().sum((1, 2)) + masks.float().sum((1, 2)) + 1e-7)
            iou = (intersection + 1e-7) / (union + 1e-7)

            dice_score += dice.mean().item()
            iou_score += iou.mean().item()

    avg_loss = total_loss / len(loader)
    pixel_acc = correct_pixels / total_pixels
    pixel_error = 1 - pixel_acc
    dice_score /= len(loader)
    iou_score /= len(loader)

    return avg_loss, pixel_acc, pixel_error, dice_score, iou_score

epochs = 31

for epoch in range(1, epochs):
    print(f"\nEpoch [{epoch}/{epochs}]")

    train_loss = train_fn(train_loader, model, optimizer, loss_fn, device)
    val_loss, pixel_acc, pixel_err, dice, iou = eval_fn(test_loader, model, loss_fn, device)

    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Pixel Acc: {pixel_acc*100:.2f}% | "
              f"Pixel Error: {pixel_err*100:.2f}% | "
              f"Dice: {dice:.4f} | iou: {iou:.4f}")
    






import matplotlib.pyplot as plt

def visualize_predictions(model, loader, device, num=5):
    model.eval()
    images_shown = 0
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds_classes = torch.argmax(preds, dim=1)

            imgs = imgs.cpu().numpy()
            masks = masks.squeeze(1).cpu().numpy()
            preds_classes = preds_classes.cpu().numpy()

            for i in range(imgs.shape[0]):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(imgs[i][0], cmap='gray')
                axs[0].set_title("Input Image")
                axs[1].imshow(masks[i], cmap='gray')
                axs[1].set_title("Ground Truth Mask")
                axs[2].imshow(preds_classes[i], cmap='gray')
                axs[2].set_title("Predicted Mask")
                for ax in axs:
                    ax.axis('off')
                plt.show()

                images_shown += 1
                if images_shown >= num:
                    return


visualize_predictions(model, test_loader, device, num=5)