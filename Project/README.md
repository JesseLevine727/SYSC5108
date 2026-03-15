# EuroSAT ResNet Baselines

This directory contains the first project experiments: training `ResNet18`, `ResNet50`, and `ViT-Small` on the EuroSAT RGB dataset with a deterministic `70/15/15` train/validation/test split.

The default training run uses `40` epochs because the initial validation-loss sweep showed the curve flattening in roughly the `34-40` epoch range.

## Dataset Layout

The training script expects the extracted dataset at:

```text
../data/EuroSAT_RGB/EuroSAT_RGB
```

## Run

From the repository root:

```bash
./.venv/bin/python Project/train_classifier.py
./.venv/bin/python Project/train_classifier.py --model resnet50
./.venv/bin/python Project/train_classifier.py --model vit_small
```

To use ImageNet-pretrained weights:

```bash
./.venv/bin/python Project/train_classifier.py --pretrained
./.venv/bin/python Project/train_classifier.py --model resnet50 --pretrained
./.venv/bin/python Project/train_classifier.py --model vit_small --pretrained
```

By default, outputs are written under model-specific directories such as:

```text
Project/outputs/resnet18_eurosat_40ep
Project/outputs/resnet50_eurosat_40ep
Project/outputs/resnet18_eurosat_40ep_pretrained
Project/outputs/resnet50_eurosat_40ep_pretrained
Project/outputs/vit_small_eurosat_40ep
```

To visualize `ViT-Small` attention maps for a saved checkpoint:

```bash
./.venv/bin/python Project/visualize_vit_attention.py
```

To batch-generate a few correct and incorrect `ViT-Small` attention maps from the test split:

```bash
./.venv/bin/python Project/batch_visualize_vit_attention.py --device cuda
```
