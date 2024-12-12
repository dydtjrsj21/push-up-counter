import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 로드
weights = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weights['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

# 이미지 읽기 및 전처리
image = cv2.imread('./person.png')
image = letterbox(image, 960, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))

if torch.cuda.is_available():
    image = image.half().to(device)

# 모델 추론
output, _ = model(image)

# 후처리
output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
with torch.no_grad():
    output = output_to_keypoint(output)

# 결과 이미지 생성
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

# 이미지 저장
output_path = './person_output.png'
cv2.imwrite(output_path, nimg)
print(f"이미지가 저장되었습니다: {output_path}")
