import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob


# ====== 配置 ======
gpu_id = 2  # 指定使用的 GPU 编号
device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 路径配置
base_path = "/data2/users/zhangxu/ISP/1_AIR/PhyDAE/dataset/MDRS-Landsat/train/"
lq_dirs = {
    "haze": os.path.join(base_path, "haze"),
    "dark": os.path.join(base_path, "dark"),
    "blur": os.path.join(base_path, "blur"),
    "noise": os.path.join(base_path, "noise"),
}
gt_dir = os.path.join(base_path, "clean")  # 所有退化类型共享同一个GT目录


# 模型配置
model_path = "/data2/users/zhangxu/BModel/blip-image-captioning-base/"
batch_size = 128  # 根据 GPU 显存调整
output_csv = "/data2/users/zhangxu/ISP/1_AIR/CyclicPrompt/csv/MDRS-Landsat_train_all_degradations.csv"


# ====== 加载模型 ======
print("加载BLIP模型...")
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model.to(device)
model.eval()


# ====== 自定义数据集 ======
class MultiDegradationDataset(Dataset):
    def __init__(self, lq_dirs, gt_dir, processor):
        """
        加载多种退化类型的数据
        Args:
            lq_dirs: dict, {退化类型: 路径}
            gt_dir: str, GT图片目录
            processor: BLIP处理器
        """
        self.processor = processor
        self.gt_dir = gt_dir
        self.samples = []  # 每个元素: (lq_path, gt_path, degradation_type)
        
        # 收集所有退化类型的图片
        for deg_type, lq_dir in lq_dirs.items():
            if not os.path.exists(lq_dir):
                print(f"⚠️ 警告: LQ目录不存在: {lq_dir}")
                continue
                
            # 获取该退化类型的所有图片
            image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
            lq_files = []
            for ext in image_extensions:
                lq_files.extend(glob.glob(os.path.join(lq_dir, '**', ext), recursive=True))
            
            print(f"找到 {len(lq_files)} 张 {deg_type} 图片")
            
            # 匹配LQ和GT
            for lq_path in lq_files:
                # 计算相对路径
                rel_path = os.path.relpath(lq_path, lq_dir)
                gt_path = os.path.join(gt_dir, rel_path)
                
                if os.path.exists(gt_path):
                    self.samples.append((lq_path, gt_path, deg_type))
                else:
                    # 尝试不同的文件名匹配策略
                    gt_path_alt = self._find_gt_path(lq_path, deg_type)
                    if gt_path_alt and os.path.exists(gt_path_alt):
                        self.samples.append((lq_path, gt_path_alt, deg_type))
                    else:
                        print(f"⚠️ 警告: 找不到 {deg_type} 对应的GT: {lq_path}")
        
        print(f"\n📊 总计收集到 {len(self.samples)} 个有效样本:")
        for deg_type in lq_dirs.keys():
            count = sum(1 for _, _, d in self.samples if d == deg_type)
            print(f"  {deg_type}: {count} 个样本")
    
    def _find_gt_path(self, lq_path, degradation_type):
        """尝试多种方式查找对应的GT路径"""
        filename = os.path.basename(lq_path)
        
        # 尝试1: 在GT目录中查找同名文件
        gt_candidates = [
            os.path.join(self.gt_dir, filename),
            os.path.join(self.gt_dir, filename.replace(f"_{degradation_type}", "")),
            os.path.join(self.gt_dir, filename.replace(f"_{degradation_type}_", "_")),
        ]
        
        for candidate in gt_candidates:
            if os.path.exists(candidate):
                return candidate
        
        # 尝试2: 通过glob查找相似文件名
        name_without_ext = os.path.splitext(filename)[0]
        pattern = os.path.join(self.gt_dir, f"{name_without_ext.split('_')[0]}*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        lq_path, gt_path, degradation = self.samples[idx]
        
        try:
            # 加载LQ图片用于生成caption
            image = Image.open(lq_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0)
        except Exception as e:
            print(f"Error loading {lq_path}: {e}")
            pixel_values = torch.zeros((3, 224, 224))
        
        return pixel_values, lq_path, gt_path, degradation


# ====== 创建数据集 ======
print("创建数据集...")
dataset = MultiDegradationDataset(lq_dirs, gt_dir, processor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# ====== 推理主流程 ======
print(f"开始处理 {len(dataset)} 个样本...")
all_data = []
batch_count = 0

with torch.no_grad():
    for batch_idx, (pixel_values_batch, lq_paths_batch, gt_paths_batch, degradations_batch) in enumerate(tqdm(dataloader, desc="生成Caption")):
        pixel_values_batch = pixel_values_batch.to(device)
        
        # 生成 caption
        generated_ids = model.generate(
            pixel_values=pixel_values_batch, 
            max_length=20,
            num_beams=5,
            early_stopping=True
        )
        captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for lq_path, gt_path, deg_type, caption in zip(lq_paths_batch, gt_paths_batch, degradations_batch, captions):
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
                "degradation": deg_type
            })
        
        batch_count += 1
        
        # 每处理10个batch打印一次进度
        if batch_count % 10 == 0:
            print(f"[进度] 已处理 {len(all_data)} 个样本")


# ====== 保存结果 ======
if all_data:
    df = pd.DataFrame(all_data)
    
    # 按退化类型排序，便于查看
    df = df.sort_values(by=['degradation', 'lq_path'])
    
    # 保存CSV
    df.to_csv(output_csv, index=False, sep="\t")
    
    print(f"\n✅ 保存成功!")
    print(f"📁 文件路径: {output_csv}")
    print(f"📊 统计信息:")
    print(f"  总样本数: {len(df)}")
    for deg_type in sorted(df['degradation'].unique()):
        count = len(df[df['degradation'] == deg_type])
        print(f"  {deg_type}: {count} 个样本")
    
    # 保存统计信息
    stats_path = output_csv.replace('.csv', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"总样本数: {len(df)}\n")
        for deg_type in sorted(df['degradation'].unique()):
            count = len(df[df['degradation'] == deg_type])
            f.write(f"{deg_type}: {count}\n")
    
    print(f"📈 统计信息保存到: {stats_path}")
else:
    print("❌ 没有处理任何图片")


# ====== 验证输出 ======
if os.path.exists(output_csv):
    print(f"\n🔍 验证生成的CSV文件:")
    df_check = pd.read_csv(output_csv, sep="\t")
    
    # 显示前几行
    print("\n前5行数据:")
    for i, row in df_check.head().iterrows():
        print(f"  {i}: [{row['degradation']}] LQ: {os.path.basename(row['lq_path'])} -> Caption: {row['caption'][:30]}...")
    
    # 验证所有路径都存在
    print("\n验证文件路径...")
    valid_count = 0
    for _, row in df_check.iterrows():
        if os.path.exists(row['lq_path']) and os.path.exists(row['gt_path']):
            valid_count += 1
    
    print(f"有效路径: {valid_count}/{len(df_check)} ({valid_count/len(df_check)*100:.1f}%)")
    
    if valid_count < len(df_check):
        print("⚠️ 警告: 部分路径无效，请检查！")