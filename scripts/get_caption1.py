import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader


# ====== 配置 ======
gpu_id = 3  # 指定使用的 GPU 编号
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

lq_path = "/data2/users/zhangxu/ISP/1_AIR/PhyDAE/dataset/MDRS-Landsat/test/blur/"
gt_path = "/data2/users/zhangxu/ISP/1_AIR/PhyDAE/dataset/MDRS-Landsat/test/clean/"  # 添加GT路径
degradation = "blur"
batch_size = 128  # 根据 GPU 显存调整
output_csv = "/data2/users/zhangxu/ISP/1_AIR/RS_recycle/csv/MDRS-Landsat_test_B_caption.csv"


# ====== 加载模型 ======
processor = BlipProcessor.from_pretrained("/data2/users/zhangxu/BModel/blip-image-captioning-base/")
model = BlipForConditionalGeneration.from_pretrained("/data2/users/zhangxu/BModel/blip-image-captioning-base/")
model.to(device)
model.eval()  # 推理模式


# ====== 自定义数据集 ======
class ImagePathDataset(Dataset):
    def __init__(self, lq_root, gt_root, processor):
        """
        同时加载LQ和GT图片路径
        Args:
            lq_root: 低质量图片根目录
            gt_root: 高质量图片根目录
            processor: BLIP处理器
        """
        self.processor = processor
        self.lq_root = lq_root
        self.gt_root = gt_root
        self.image_pairs = self._load_image_pairs()
        
        print(f"找到 {len(self.image_pairs)} 张图片")
    
    def _load_image_pairs(self):
        """加载匹配的LQ-GT图片对"""
        image_pairs = []
        
        # 遍历LQ目录
        for root, _, files in os.walk(self.lq_root):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                    
                lq_path = os.path.join(root, file)
                # 计算相对路径
                rel_path = os.path.relpath(lq_path, self.lq_root)
                # 构建对应的GT路径
                gt_path = os.path.join(self.gt_root, rel_path)
                
                # 检查GT文件是否存在
                if os.path.exists(gt_path):
                    image_pairs.append((lq_path, gt_path))
                else:
                    print(f"⚠️ 警告: GT文件不存在: {gt_path}")
                    # 可以选择记录或跳过
                    
        return image_pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        lq_path, gt_path = self.image_pairs[idx]
        
        try:
            # 加载LQ图片用于生成caption
            image = Image.open(lq_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0)  # 移除 batch 维度
        except Exception as e:
            print(f"Error loading {lq_path}: {e}")
            pixel_values = torch.zeros((3, 224, 224))  # BLIP 默认输入尺寸
        
        return pixel_values, lq_path, gt_path


# ====== 创建数据集 ======
dataset = ImagePathDataset(lq_path, gt_path, processor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ====== 推理主流程 ======
all_data = []

with torch.no_grad():
    for batch_idx, (pixel_values_batch, lq_paths_batch, gt_paths_batch) in enumerate(dataloader):
        pixel_values_batch = pixel_values_batch.to(device)

        # 生成 caption
        generated_ids = model.generate(pixel_values=pixel_values_batch, max_length=20)
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for lq_path, gt_path, caption in zip(lq_paths_batch, gt_paths_batch, captions):
            # 验证文件存在
            if not os.path.exists(lq_path):
                print(f"❌ 错误: LQ文件不存在: {lq_path}")
                continue
            if not os.path.exists(gt_path):
                print(f"❌ 错误: GT文件不存在: {gt_path}")
                continue

            all_data.append({
                "gt_path": gt_path,
                "lq_path": lq_path,
                "caption": caption,
                "degradation": degradation
            })

            # 打印信息
            lq_filename = os.path.basename(lq_path)
            gt_filename = os.path.basename(gt_path)
            print(f"[{batch_idx+1:03d}] LQ: {lq_filename} -> GT: {gt_filename} -> Caption: {caption}")

        print(f"[Batch {batch_idx + 1}/{len(dataloader)}] Processed {len(lq_paths_batch)} images")


# ====== 保存结果 ======
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False, sep="\t")
    print(f"✅ 保存成功! 共处理 {len(df)} 张图片")
    print(f"📁 保存路径: {output_csv}")
else:
    print("❌ 没有处理任何图片，请检查路径设置")


# ====== 验证输出 ======
# 可选：验证生成的CSV文件
if os.path.exists(output_csv):
    df_check = pd.read_csv(output_csv, sep="\t")
    print(f"\n📊 CSV文件验证:")
    print(f"总行数: {len(df_check)}")
    print(f"列名: {list(df_check.columns)}")
    print("\n前5行数据:")
    print(df_check.head())
    
    # 检查路径是否存在
    valid_paths = 0
    for _, row in df_check.iterrows():
        if os.path.exists(row['lq_path']) and os.path.exists(row['gt_path']):
            valid_paths += 1
    print(f"有效路径: {valid_paths}/{len(df_check)}")