from model_resnet import *
from dataloader import *
from torchvision import transforms

test_path = 'data/test_galaxy/'
model_name = 'resnet18'
pretrain_check = False

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
    model = resnet18(pretrained=pretrain_check).cuda()
elif model_name == 'resnet34':
    model = resnet34(pretrained=pretrain_check).cuda()
elif model_name == 'resnet50':
    model = resnet50(pretrained=pretrain_check).cuda()
elif model_name == 'resnet101':
    model = resnet101(pretrained=pretrain_check).cuda()
elif model_name == 'resnet152':
    model = resnet152(pretrained=pretrain_check).cuda()
elif model_name == 'resnext50_32x4d':
    model = resnext50_32x4d(pretrained=pretrain_check).cuda()
elif model_name == 'resnext101_32x8d':
    model = resnext101_32x8d(pretrained=pretrain_check).cuda()
elif model_name == 'wide_resnet50_2':
    model = wide_resnet50_2(pretrained=pretrain_check).cuda()
elif model_name == 'wide_resnet101_2':
    model = wide_resnet101_2(pretrained=pretrain_check).cuda()

model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(r'C:\Users\a\OneDrive - 고려대학교\toyproject\딥러닝\checkpoint\resnet18_3\resnet18_21.pt')['state_dict'])
model.eval()



n=0
y_true = []
y_pred = []
for i in test_loader:
    y_true.append(i[1])
    y_pred.append(torch.argmax(model(i[0])).cpu())
    if (torch.argmax(model(i[0])).cpu()== i[1]):
        n+=1

print('Accuracy : ', n/len(test_loader)*100, '%')

from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)