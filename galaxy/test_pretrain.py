from model_resnet_pretrained import *
from dataloader import *
from torchvision import transforms
from torch import nn

test_path = 'data/test_galaxy/'
model_name = 'resnet101'
pretrain_check = True

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
# transformList.append(transforms.ToPILImage())
# transformList.append(transforms.Resize(imgtransResize))
# transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformVal = transforms.Compose(transformList)

# Test

test_dataset = GalD(test_path, transform=transformVal)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if model_name == 'resnet18':
    model = resnet18(num_classes=1000, pretrained=None).cuda()
elif model_name == 'resnet34':
    model = resnet34(num_classes=1000, pretrained='imagenet').cuda()
elif model_name == 'resnet50':
    model = resnet50(num_classes=1000, pretrained='imagenet').cuda()
elif model_name == 'resnet101':
    model = resnet101(num_classes=1000, pretrained='imagenet').cuda()
elif model_name == 'resnet152':
    model = resnet152(num_classes=1000, pretrained='imagenet').cuda()


dim_feats = model.last_linear.in_features # =2048
nb_classes = 3
model.last_linear = nn.Linear(dim_feats, nb_classes).cuda()


model.load_state_dict(torch.load(r'C:\Users\a\OneDrive - 고려대학교\toyproject\딥러닝\checkpoint\resnet101\_15.pt')['state_dict'])
model.eval()


n=0
y_true = []
y_pred = []
for i in test_loader:
    y_true.append(i[1])
    y_pred.append(torch.argmax(model(i[0].cuda())).cpu())
    if (torch.argmax(model(i[0].cuda())).cpu()== i[1]):
        n+=1

print('Accuracy : ', n/len(test_loader)*100, '%')

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)