# Magicremover: Tuning-free Text-guided Image inpainting with Diffusion Models

## Quick Start

### Installation

```bash
conda create -n remover python=3.10 -y
conda activate remover
pip install -r requirements.txt
```

### Test Our Method

```bash
python main.py --img_path="xxx" --prompts="a photo of a dog ." --idx=5 --t_ratio=0.85
```

---

## Test on COCO

### Data Preparation

1. Download the [COCO 2017 val images](http://images.cocodataset.org/zips/val2017.zip)
2. Download the [COCO 2017 annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
3. Organize your data as follows:

```
data/
├── val2017/
│   ├── [id].jpg
│   └── ...
└── annotations/
    ├── instances_val2017.json
    └── ...
```

---

### Null-Text-Inversion

1. Generate null text embeddings:

    ```bash
    python null_text_inversion_coco.py --data_dir="./data" --task="embeddings_generation"
    ```

2. Object removal:

    ```bash
    python null_text_inversion_coco.py --data_dir="./data" --task="object_removal"
    ```

---

### Evaluation

- **Calculate FID:**
    ```bash
    python evaluation/evaluate.py --task=fid --coco_output_path="./null_text_embeddings_coco/images"
    ```

- **Calculate UIDS:**
    ```bash
    python evaluation/calculate_uids.py --coco_output_path="./null_text_embeddings_coco/images"
    ```

- **Calculate CLIP similarity:**
    ```bash
    python evaluation/evaluate.py --task=cs --coco_output_path="./null_text_embeddings_coco/images" --coco_labels_file="data/annotations/instances_val2017.json"
    ```