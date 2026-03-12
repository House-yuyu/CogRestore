# CogRestore

## 📦 Datasets

We evaluate CogRestore on two remote sensing image restoration datasets. All datasets can be downloaded from the links below.

### Download Links

| Dataset | Description | Baidu Netdisk |
|---------|-------------|:-------------:|
| iSAID2  | Synthetic composite degradation (noise + low-light) | [Download](https://pan.baidu.com/s/xxxxx) (Code: `xxxx`) |
| RealRS  | Real-world degradation (no ground-truth) | [Download](https://pan.baidu.com/s/xxxxx) (Code: `xxxx`) |

### Dataset Details

| Dataset | Train | Val | Test | Image Size | Degradation Type |
|---------|:-----:|:---:|:----:|:----------:|:----------------:|
| iSAID2  | x,xxx | xxx | xxx  | 512×512    | Composite (noise + low-light) |
| RealRS  | —     | —   | xxx  | 512×512    | Real-world |

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
