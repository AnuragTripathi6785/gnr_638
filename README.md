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

## How to Reproduce

Easiest way is to use the provided Colab notebook (`YOLOv1_Colab.ipynb`), which handles setup automatically. Steps below are for running locally.

### 1. Clone and install dependencies

```bash
git clone https://github.com/AnuragTripathi6785/gnr_638.git
cd gnr_638
pip install torch torchvision pillow numpy matplotlib
```

### 2. Download the dataset

Download the Hocmay dataset from Roboflow in YOLO format:
https://universe.roboflow.com/suyogs-workspace-voqgn/hocmay1-39isr

Place it as:
```
data/
  train/images/   train/labels/
  valid/images/   valid/labels/
  test/images/    test/labels/
```

Update `DATA_ROOT` in `config.py` if needed (defaults to `./data`).

### 3. Download pretrained backbone (optional but recommended)

The original YOLOv1 paper pretrains the first 20 conv layers on ImageNet classification before fine-tuning for detection. We follow the same approach using the `extraction.conv.weights` file released by Redmon.

```bash
wget https://pjreddie.com/media/files/extraction.conv.weights
```

Load it before training by calling `load_darknet_conv_weights(model, "extraction.conv.weights")` in `train.py` (see the Colab notebook for exact usage).

### 4. Train

```bash
python train.py
```

Trains for 135 epochs. Best checkpoint saved to `checkpoints/yolov1_best.pth`.

### 5. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/yolov1_best.pth --split valid
```

### 6. Run detection on images

```bash
python detect.py --checkpoint checkpoints/yolov1_best.pth --img-dir data/test/images --output-dir detections/
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
