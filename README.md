# Trustworthy Knowledge Distillation via Anchor-Guided Distribution Learning

This repo is for reproducing the CIFAR-100 experimental results in our paper Trustworthy Knowledge Distillation via Anchor-Guided Distribution Learning.


To perform knowledge transfer from Resnet32x4 to Resnet8x4 on CIFAR-100 using AKD, run:

```sh
python train_cifar_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill AKD --model_t resnet32x4 --model_s resnet8x4
```
