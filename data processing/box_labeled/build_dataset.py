import os
import json
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# from general_inference import convert

class RefCOCODataset(Dataset):
    def __init__(self,
                 coco_img_dir,
                 instances_json,
                 ref_pickle_file,
                 transform=None,
                 use_convert=False):
        """
        Args:
            coco_img_dir: COCO 图片所在目录（例如 train2014 文件夹路径）
            instances_json: COCO 标注文件路径（instances.json）
            ref_pickle_file: 指代表达 pickle 文件路径（例如 refs(google).p 或 refs(umd).p）
            transform: 图像预处理函数（例如 torchvision.transforms）
            use_convert: 是否调用 convert 函数对图像进行预处理
        """
        # 加载 COCO 标注文件
        with open(instances_json, 'r') as f:
            coco_data = json.load(f)

        # 构建 image_id -> 文件名 的映射
        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        # 构建 annotation_id -> 标注信息 的映射
        self.ann_id_to_ann = {ann['id']: ann for ann in coco_data['annotations']}

        self.coco_img_dir = coco_img_dir

        # 加载指代表达 pickle 文件
        with open(ref_pickle_file, 'rb') as f:
            self.refs = pickle.load(f)

        # 调试：打印第一个 item 的键和原始 sentences 数据
        if len(self.refs) > 0:
            first_item = self.refs[0]
            print("Pickle 文件中第一个 item 的 keys:", first_item.keys())
            sentences_raw = first_item.get('sentences')
            print("Raw sentences:", sentences_raw, type(sentences_raw))

        # 构建训练样本列表
        self.samples = []
        for item in self.refs:
            image_id = item.get('image_id')
            ann_id = item.get('ann_id')
            if image_id is None or ann_id is None:
                continue

            sentences = item.get('sentences', ())
            # 直接取第一个元素，并提取其中 "raw" 字段（如果存在）
            if isinstance(sentences, (list, tuple)) and len(sentences) > 0:
                first_element = sentences[0]
                if isinstance(first_element, dict) and 'raw' in first_element:
                    ref_text = first_element['raw']
                elif isinstance(first_element, str):
                    ref_text = first_element
                else:
                    ref_text = str(first_element)
            else:
                ref_text = str(sentences)

            if image_id in self.image_id_to_filename and ann_id in self.ann_id_to_ann:
                image_filename = self.image_id_to_filename[image_id]
                ann = self.ann_id_to_ann[ann_id]
                bbox = ann.get('bbox', None)  # 格式通常为 [x, y, width, height]
                sample = {
                    'image_path': os.path.join(self.coco_img_dir, image_filename),
                    'ref': ref_text,
                    'bbox': bbox
                }
                self.samples.append(sample)

        # 如果没有传入 transform，使用默认的 ToTensor
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.use_convert = use_convert

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 加载原始图像
        image = Image.open(sample['image_path']).convert('RGB')
        # 如果需要调用 convert 函数，则处理图像
        if self.use_convert:
            image = convert(image)
        # 应用 transform 预处理
        if self.transform:
            image = self.transform(image)
        return image, sample['ref'], sample['bbox']


import json


def save_dataset(dataset, save_path):
    """
    保存 dataset.samples 到 JSON 文件

    Args:
        dataset: RefCOCODataset 对象，其 samples 属性是待保存的数据列表
        save_path: 保存的文件路径，例如 "saved_dataset.json"
    """
    # 拷贝一份数据，确保所有非 JSON 序列化的数据都做了转换
    samples_to_save = []
    for sample in dataset.samples:
        sample_copy = sample.copy()
        # 如果 bbox 是 tensor 或其它不可直接序列化的对象，转换为 list
        bbox = sample_copy.get('bbox')
        if bbox is not None:
            if isinstance(bbox, (list, tuple)):
                sample_copy['bbox'] = list(bbox)
            else:
                try:
                    sample_copy['bbox'] = bbox.tolist()
                except Exception:
                    sample_copy['bbox'] = str(bbox)
        samples_to_save.append(sample_copy)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(samples_to_save, f, ensure_ascii=False, indent=4)
    print(f"Dataset 已保存到 {save_path}")


    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(samples_to_save, f, ensure_ascii=False, indent=4)
    print(f"Dataset 已保存到 {save_path}")