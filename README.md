# GNR638 Assignment 3 — YOLOv1

Reimplementation of **You Only Look Once** (Redmon et al., CVPR 2016) from scratch in PyTorch, compared against the original Darknet C implementation. Both trained on the Hocmay dataset (6 classes).

## Structure

```
config.py       hyperparameters and paths
model.py        YOLOv1 network (24 conv + 2 FC layers)
loss.py         multi-part MSE loss (Eq. 3 from paper)
dataset.py      data loader with augmentation
train.py        training loop
evaluate.py     mAP evaluation
detect.py       inference on images
pretrained.py   load Darknet .weights backbone
visualize.py    grid and GT box visualization
```

## Dataset

Hocmay dataset from Roboflow — 391 train / 112 val / 56 test images, 6 classes: bike, car, cat, cellphone, dog, person.

Expected folder layout:
```
data/
  train/images/  train/labels/
  valid/images/  valid/labels/
  test/images/   test/labels/
```

## Training

```bash
python train.py
```

Trains for 135 epochs with SGD (momentum 0.9, weight decay 5e-4). Checkpoints saved to `checkpoints/`.

To load the pretrained ImageNet backbone before training:
```python
from pretrained import load_darknet_conv_weights
model = load_darknet_conv_weights(model, "extraction.conv.weights")
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/yolov1_best.pth --split valid
```

## Results

| Implementation | mAP@0.5 | Best Epoch | Final Loss |
|---|---|---|---|
| PyTorch (scratch) | 0.0521 | 80 | 4.76 (val) |
| Darknet (official) | N/A* | 945 iters | 1.27 (train) |

*Darknet mAP eval output not captured in Colab environment.

Per-class AP (PyTorch): dog 0.292, person 0.020, others 0.000.

## References

- Redmon et al., *You Only Look Once*, CVPR 2016
- [Darknet](https://github.com/pjreddie/darknet) by Joseph Redmon
- [Hocmay dataset](https://universe.roboflow.com/suyogs-workspace-voqgn/hocmay1-39isr) on Roboflow
