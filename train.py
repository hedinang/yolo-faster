import cv2
import torch
import torchvision
import os
import math
import numpy as np
from torchvision.transforms import transforms as T
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/log')
device = torch.device('cuda')

backbone = torchvision.models.vgg16(pretrained=False).features
backbone.out_channels = 512
anchor_sizes = ((8, 16, 32, 64, 128, 256, 512),)
aspect_ratios = ((1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/math.sqrt(2), 1,
                  2, math.sqrt(2), 3, 4, 5, 6, 7, 8),)
anchor_generator = AnchorGenerator(
    sizes=anchor_sizes, aspect_ratios=aspect_ratios)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=6,
                   box_score_thresh=0.5,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler,
                   box_detections_per_img=20)

model.to(device)
# model.load_state_dict(torch.load('d.pth'))


class DocDataset(torch.utils.data.Dataset):
    def __init__(self, root_url='/home/dung/Data/doc', scale=(800, 1000)):
        self.root_url = root_url
        self.scaleW, self.scaleH = scale
        self.all_file = os.listdir(self.root_url)
        self.all_file.remove('classes.txt')
        self.imgs = []
        self.txts = []
        for f in self.all_file:
            if '.txt' in f:
                self.imgs.append(
                    '{}/{}.png'.format(self.root_url, f.split('.txt')[0]))
                self.txts.append('{}/{}'.format(self.root_url, f))
        self.transforms = T.Compose([
            T.Resize(scale),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        img = np.array(img)
        img = img[:, :, :3]
        truth = cv2.resize(img, (self.scaleH, self.scaleW))
        img = Image.fromarray(img)
        img = self.transforms(img)
        f = open(self.txts[idx], 'r')
        boxes = []
        labels = []
        for line in f.readlines():
            label, x, y, w, h = line.splitlines()[0].split(' ')
            label, x, y, w, h = int(label)+1, float(
                x), float(y), float(w), float(h)
            x0, y0, x1, y1 = (x-w/2) * self.scaleW, (y-h/2) * \
                self.scaleH, (x+w/2)*self.scaleW, (y+h/2)*self.scaleH
            boxes.append((x0, y0, x1, y1))
            labels.append(label)
        targets = {
            'boxes': torch.tensor(boxes, dtype=torch.float64),
            'labels': torch.tensor(labels, dtype=torch.int64)

        }

        return img, targets, self.imgs[idx], truth


root_url = '/home/dung/Project/Test/faster-torch/output.txt'
dataset = DocDataset()

fontScale = 1
color = (255, 0, 0)
thickness = 1
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
step = 0
for i in range(500):
    print('Epoch {}\n'.format(i))
    for j, (images, targets, path, truth) in enumerate(data_loader):
        images = images.to(device)
        images = list(image for image in images)
        b = []
        boxes = []
        labels = []
        for k in range(len(targets['boxes'])):
            a = {}
            a['boxes'] = targets['boxes'][k].to(device)
            a['labels'] = targets['labels'][k].to(device)
            boxes = a['boxes']
            labels = a['labels']
            b.append(a)
        output = model(images, b)
        losses = sum(loss for loss in output.values())
        writer.add_scalar('loss_classifier',
                          output['loss_classifier'].item(), step)
        writer.add_scalar('loss_box_reg', output['loss_box_reg'].item(), step)
        writer.add_scalar('loss_objectness',
                          output['loss_objectness'].item(), step)
        writer.add_scalar('loss_rpn_box_reg',
                          output['loss_rpn_box_reg'].item())
        # ground truth
        # boxes = boxes.detach().cpu().numpy()
        # labels = labels.detach().cpu().numpy()
        # truth = truth[0].detach().numpy()
        # refine = truth
        # for k in range(len(boxes)):
        #     x0, y0, x1, y1 = boxes[i]

        #     cv2.rectangle(truth, (int(x0), int(y0)), (int(x1), int(y1)),
        #                   (100, 200, 150), 1, 1)
        #     cv2.putText(truth, str(labels[i]), (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
        #                 color, thickness, cv2.LINE_AA)

        # boxes = output[0]['boxes']
        # scores = output[0]['scores']
        # labels = output[0]['labels'].detach().cpu().numpy()

        # for i, score in enumerate(scores):
        #     if score > 0.5:
        #         x0, y0, x1, y1 = boxes[i]
        #         cv2.rectangle(refine, (x0, y0), (x1, y1),
        #                       (100, 200, 150), 1, 1)
        #         cv2.putText(refine, str(labels[i]), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fontScale,u
        #                     color, thickness, cv2.LINE_AA)
        # combine = torch.concatenate(images[0],)
        # writer.add_image('image',)
        if j % 100 == 0:
            print('Step {} -- loss_classifier = {} -- loss_box_reg = {} -- loss_objectness = {} -- loss_rpn_box_reg = {}\n'.format(j,
                                                                                                                                   output['loss_classifier'].item(), output['loss_box_reg'].item(), output['loss_objectness'].item(), output['loss_rpn_box_reg'].item()))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        step += 1
    if i % 50 == 0:
        torch.save(model.state_dict(), 'document.pth')
print('done')
