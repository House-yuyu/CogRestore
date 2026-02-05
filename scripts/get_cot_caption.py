import torch
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import re
import os

class RSImageCOTDescriber:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_think_mode: bool = False):
        """
        初始化CoT描述生成器
        """
        self.device = device
        self.use_think_mode = use_think_mode
        
        # 初始化模型和处理器
        self.model, self.processor = self._load_multimodal_model(model_name)
        
        # CoT提示模板
        self.cot_templates = self._init_cot_templates()
        
        # 领域词汇表
        self.land_cover_vocab = {
            "urban": ["building", "road", "bridge", "highway", "residential area", "commercial area"],
            "vegetation": ["forest", "farmland", "crop field", "grassland", "orchard", "park"],
            "water": ["river", "lake", "reservoir", "coastline", "sea", "pond"],
            "barren": ["desert", "bare land", "mountain", "rocky area", "construction site"],
            "mixed": ["suburban", "rural area", "industrial area", "port"]
        }
        
        self.degradation_vocab = {
            "blur": ["motion blur", "defocus blur", "atmospheric blur", "gaussian blur"],
            "noise": ["gaussian noise", "salt-and-pepper noise", "quantization noise"],
            "compression": ["JPEG artifacts", "blocking artifacts", "banding artifacts"],
            "weather": ["haze", "cloud cover", "fog"],
            "sensor": ["sensor noise", "striping noise"]
        }
    
    def _load_multimodal_model(self, model_name: str):
        """加载Qwen2.5-VL多模态模型"""
        try:
            from transformers import AutoProcessor
            
            print(f"加载Qwen2.5-VL多模态模型: {model_name}")
            
            # 使用Qwen2_5_VLForConditionalGeneration
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                print("✓ 使用Qwen2_5_VLForConditionalGeneration")
                model_class = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                print("⚠ Qwen2_5_VLForConditionalGeneration不可用，尝试AutoModel")
                from transformers import AutoModel
                model_class = AutoModel
            
            # ✅ 修改开始：添加GPU选择逻辑
            if "cuda" in self.device:
                # 解析GPU ID
                if ":" in self.device:
                    gpu_id = int(self.device.split(":")[1])
                else:
                    gpu_id = 0
                
                # 只使用指定GPU
                device_map = {"": gpu_id}
                print(f"模型将加载到 GPU {gpu_id}")
            else:
                device_map = None
            # ✅ 修改结束
            
            model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=device_map,  # ✅ 使用指定的device_map
                trust_remote_code=True
            )
            
            # 加载处理器
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # 如果使用device_map="auto"，模型已分配到设备
            # 否则手动移动
            if "cuda" in self.device and not hasattr(model, "hf_device_map"):
                model = model.to(self.device)
            
            model.eval()
            print("✅ Qwen2.5-VL多模态模型加载成功!")
            
            return model, processor
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def _init_cot_templates(self) -> Dict:
        """初始化CoT提示模板"""
        return {
            "cot_system": """你是一个专业的遥感图像分析专家。""",
            
            "cot_analysis": """请对这张遥感图像进行逐步分析，并生成用于图像复原的文本描述。

请严格按以下步骤进行推理：

## 步骤1：地物识别与场景理解
1. 主要地物类别：
   - 城市区域：[建筑密度，道路网络，功能区分布]
   - 植被区域：[森林/农田类型，植被密度]
   - 水体区域：[水体类型，边界清晰度]
   - 其他地物：[具体描述]

2. 空间布局与结构：
   - 整体布局特征：[规则/不规则]
   - 显著空间关系：[相对位置]
   - 纹理模式：[均匀/异质]

## 步骤2：退化分析与质量评估
1. 退化类型识别：
   - 模糊类型：[{blur_types}]
   - 噪声类型：[{noise_types}]
   - 压缩伪影：[{compression_types}]
   - 天气影响：[{weather_types}]
   - 传感器问题：[{sensor_types}]

2. 退化程度评估：
   - 总体严重程度：[轻度/中度/严重]
   - 对识别的影响：[哪些地物难以辨认]

## 步骤3：信息确定性分析
1. 高度确定的信息：
   - [明确可识别的内容]
2. 中等确定的信息：
   - [部分可识别的内容]
3. 不确定的信息：
   - [推测的内容]

## 步骤4：多粒度文本描述生成
基于以上分析，生成三种描述：

1. 【精细描述】（用于细节复原）：
[具体描述地物、空间关系、纹理细节]

2. 【主题描述】（用于语义复原）：
[概括图像的核心主题]

3. 【属性描述】（用于风格复原）：
[描述图像的质量属性]""",
            
            "direct_generation": "请描述这张遥感图像。",
            "simple_cot": "请逐步描述这张遥感图像。"
        }
    
    def _format_cot_prompt(self, degradation_focus: Optional[str] = None) -> str:
        """格式化CoT提示"""
        template = self.cot_templates["cot_analysis"]
        
        # 填充退化类型
        formatted = template.format(
            blur_types=", ".join(self.degradation_vocab["blur"]),
            noise_types=", ".join(self.degradation_vocab["noise"]),
            compression_types=", ".join(self.degradation_vocab["compression"]),
            weather_types=", ".join(self.degradation_vocab["weather"]),
            sensor_types=", ".join(self.degradation_vocab["sensor"])
        )
        
        if degradation_focus:
            formatted += f"\n注意：图像主要受到{degradation_focus}影响。"
        
        return formatted
    
    def load_image(self, image_path: str) -> Image.Image:
        """加载图像"""
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path, stream=True, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    
    def generate_descriptions(self, 
                             image_path: str,
                             method: str = "cot",
                             degradation_focus: Optional[str] = None,
                             max_new_tokens: int = 1024,
                             temperature: float = 0.7) -> Dict[str, str]:
        """生成描述"""
        image = self.load_image(image_path)
        
        # 获取提示
        if method == "direct":
            prompt = self.cot_templates["direct_generation"]
        elif method == "simple_cot":
            prompt = self.cot_templates["simple_cot"]
        else:
            prompt = self._format_cot_prompt(degradation_focus)
        
        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # 应用聊天模板
        try:
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except AttributeError:
            # 回退方案
            text = f"<|im_start|>user\n"
            text += f"<|image|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 处理输入
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        # if not hasattr(self.model, "hf_device_map"):
        #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}


        # 生成
        with torch.no_grad():
            # 检查模型是否有generate方法
            if hasattr(self.model, 'generate'):
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            else:
                # 如果模型没有generate方法，手动实现
                outputs = self.model(**inputs)
                generated_ids = torch.argmax(outputs.logits, dim=-1)
        
        # 解码
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        # 解析
        if method == "cot":
            return self._extract_descriptions(generated_text)
        else:
            return {
                "fine_grained": generated_text[:300],
                "thematic": "",
                "attribute": "",
                "full_cot": generated_text
            }
    
    def _extract_descriptions(self, cot_output: str) -> Dict[str, str]:
        """提取三种描述"""
        descriptions = {
            "fine_grained": "",
            "thematic": "",
            "attribute": "",
            "full_cot": cot_output
        }
        
        # 使用正则表达式提取
        patterns = {
            "fine_grained": r"【精细描述】[：:]\s*(.*?)(?=\n【|$)",
            "thematic": r"【主题描述】[：:]\s*(.*?)(?=\n【|$)",
            "attribute": r"【属性描述】[：:]\s*(.*?)(?=\n【|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, cot_output, re.DOTALL)
            if match:
                descriptions[key] = match.group(1).strip()[:300]
        
        return descriptions
    
    def analyze_and_export(self, 
                          image_path: str,
                          output_json: Optional[str] = None) -> Dict:
        """分析并导出结果"""
        print(f"分析图像: {image_path}")
        print("-" * 50)
        
        cot_result = self.generate_descriptions(
            image_path, 
            method="cot", 
            max_new_tokens=1200
        )
        
        direct_result = self.generate_descriptions(
            image_path, 
            method="direct", 
            max_new_tokens=500
        )
        
        result = {
            "image_path": image_path,
            "generation_methods": {
                "cot": cot_result,
                "direct": direct_result
            },
            "metadata": {
                "model": "Qwen2.5-VL-7B-Instruct",
                "method": "degradation_aware_cot",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self._print_summary(result)
        
        if output_json:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_json}")
        
        return result
    
    def _print_summary(self, result: Dict):
        """打印摘要"""
        print("\n" + "="*60)
        print("分析结果摘要")
        print("="*60)
        
        cot_desc = result["generation_methods"]["cot"]
        direct_desc = result["generation_methods"]["direct"]
        
        print("\n1. 精细描述对比:")
        print(f"CoT方法: {cot_desc.get('fine_grained', '')[:200]}...")
        print(f"\n直接生成: {direct_desc.get('fine_grained', '')[:200]}...")
        
        print("\n2. CoT特有描述:")
        if cot_desc.get("thematic"):
            print(f"主题描述: {cot_desc['thematic']}")
        if cot_desc.get("attribute"):
            print(f"属性描述: {cot_desc['attribute']}")

# 主程序
if __name__ == "__main__":
    print("初始化Qwen2.5-VL CoT描述生成器...")
    
    # 注意：这里我们使用正确的加载方法
    describer = RSImageCOTDescriber(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda:0",
        use_think_mode=False
    )
    
    # 测试图像
    test_image = "/data2/users/zhangxu/ISP/1_AIR/PhyDAE/dataset/MD_RRSHID/train/blur/2.png"
    
    if os.path.exists(test_image):
        result = describer.analyze_and_export(
            test_image,
            output_json="cot_analysis_result.json"
        )
        
        print("\n" + "="*60)
        print("✅ 分析完成!")
        print("="*60)
    else:
        print(f"测试图像不存在: {test_image}")
        print("请修改为你的图像路径")