# CogRestore

## 📦 Datasets

We evaluate CogRestore on two remote sensing image restoration datasets. All datasets can be downloaded from the links below.

### Download Links

| Dataset | Description | Baidu Netdisk |
|---------|-------------|:-------------:|
| iSAID2  | Synthetic composite degradation (noise & dark) | [Download](https://pan.baidu.com/s/xxxxx) (Code: `xxxx`) |
| RealRS  | Real-world degradation (no ground-truth) | [Download](https://pan.baidu.com/s/xxxxx) (Code: `xxxx`) |

### Dataset Details

| Dataset | Train | Val | Test | Image Size | Degradation Type |
|---------|:-----:|:---:|:----:|:----------:|:----------------:|
| iSAID2  | x,xxx | xxx | xxx  | 512×512    | Composite (noise & dark) |
| RealRS  | —     | —   | xxx  | 512×512    | Real-world |
| MD-RSID | 3,200     | 400   | 400  | 256×256    | noise + blur + haze + dark |

### Dataset Structure

After downloading, please organize the datasets as follows:
data/
├── iSAID2/
│ ├── train/
│ │ ├── input/ # degraded images
│ │ └── target/ # ground-truth images
│ ├── val/
│ │ ├── input/
│ │ └── target/
│ └── test/
│ ├── input/
│ └── target/
├── RealRS/
│ └── test/
│ ├── low_light/
│ ├── noise/
│ ├── haze/
│ └── rain/


### iSAID2

The iSAID2 dataset is constructed based on the [iSAID](https://captain-whu.github.io/iSAID/) dataset. Each degraded image simultaneously suffers from **low-light** and **noise** degradations, simulating composite degradation conditions commonly encountered in real-world remote sensing scenarios.

### RealRS

The RealRS dataset contains real-world degraded remote sensing images covering four degradation scenarios: **low-light**, **noise**, **haze**, and **rain**. Since no ground-truth references are available, this dataset is used for **qualitative evaluation** only.

> **Note:** If the Baidu Netdisk links expire, please open an issue and we will update them promptly.

## 🔗 Citation

If you find our datasets useful, please consider citing:

```bibtex
@article{xxx2025cogrestore,
  title={CogRestore: xxxxxxxxxx},
  author={xxx},
  journal={xxx},
  year={2025}
}
