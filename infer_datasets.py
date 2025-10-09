import argparse
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.transforms import F
from tqdm import tqdm
from huggingface_hub import snapshot_download

from module.pipeline_fastfit import FastFitPipeline
from parse_utils.automasker import cloth_agnostic_mask, multi_ref_cloth_agnostic_mask
from parse_utils import DWposeDetector

# --- Helper Function ---


class PreprocessingChecker:
    """检查并生成缺失的dwpose预处理文件"""
    
    def __init__(self, util_model_path: str = "Models/Human-Toolkit", device: str = None):
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.util_model_path = util_model_path
        
        # 下载模型如果不存在
        if not os.path.exists(util_model_path):
            os.makedirs(util_model_path, exist_ok=True)
            snapshot_download(
                repo_id="zhengchong/Human-Toolkit",
                local_dir=util_model_path,
                local_dir_use_symlinks=False
            )
        
        # 初始化dwpose检测器
        self.dwpose_detector = DWposeDetector(
            pretrained_model_name_or_path=os.path.join(util_model_path, "DWPose"), 
            device='cpu'
        )
    
    def check_and_generate_dwpose(self, person_path: Path, dwpose_path: Path) -> bool:
        """检查并生成dwpose文件"""
        if dwpose_path.exists():
            return True
        
        try:
            # 确保输出目录存在
            dwpose_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 加载人物图像
            person_img = Image.open(person_path).convert("RGB")
            
            # 生成dwpose
            dwpose_img = self.dwpose_detector(person_img)
            if isinstance(dwpose_img, Image.Image):
                dwpose_img.save(dwpose_path)
                return True
            else:
                print(f"Failed to generate dwpose for {person_path}")
                return False
        except Exception as e:
            print(f"Error generating dwpose for {person_path}: {e}")
            return False
    
    def check_all_dwpose_files(self, data_list: list, data_dir: str) -> None:
        """检查并生成所有缺失的dwpose文件"""
        print("Checking dwpose files...")
        missing_count = 0
        total_count = 0
        
        for sample in tqdm(data_list, desc="Checking dwpose files"):
            root = Path(data_dir)
            person_path = root / sample["person"]
            
            # 根据数据集类型确定dwpose文件路径
            if "annotations" in sample["person"]:
                # DressCode-MR数据集
                dwpose_file = (
                    sample["person"].replace("person", "annotations/dwpose").rsplit(".", 1)[0]
                    + ".png"
                )
            elif "person" in sample["person"]:
                # DressCode数据集
                dwpose_file = (
                    sample["person"].replace("person", "dwpose").rsplit(".", 1)[0] + ".png"
                )
            elif "image" in sample["person"]:
                # VitonHD数据集
                dwpose_file = (
                    sample["person"].replace("image", "dwpose").rsplit(".", 1)[0] + ".png"
                )
            else:
                continue
            
            dwpose_path = root / dwpose_file
            total_count += 1
            
            if not dwpose_path.exists():
                missing_count += 1
                success = self.check_and_generate_dwpose(person_path, dwpose_path)
                if success:
                    print(f"Generated dwpose: {dwpose_path}")
                else:
                    print(f"Failed to generate dwpose: {dwpose_path}")
        
        print(f"Dwpose check completed. Total: {total_count}, Missing: {missing_count}")


def center_crop_max_area_by_aspect_ratio(
    img: Image.Image, target_ratio: float
) -> Image.Image:
    """
    Crops the image to the target aspect ratio, centered, preserving the maximum possible area.

    Args:
        img (Image.Image): The input PIL Image.
        target_ratio (float): The target aspect ratio (width / height).

    Returns:
        Image.Image: The cropped PIL Image.
    """
    width, height = img.size
    original_ratio = width / height

    if original_ratio > target_ratio:
        # Original is wider than target: crop width
        new_width = int(height * target_ratio)
        new_height = height
    else:
        # Original is taller than or equal to target: crop height
        new_width = width
        new_height = int(width / target_ratio)

    left = (width - new_width) // 2
    upper = (height - new_height) // 2
    right = left + new_width
    lower = upper + new_height

    return img.crop((left, upper, right, lower))


# --- Dataset ---


class DressCodeMRDataset(Dataset):
    """
    A PyTorch Dataset for the DressCode-MR (Multi-Reference) dataset.

    This class handles loading a person's image, multiple reference clothing items,
    and corresponding masks and poses for virtual try-on tasks.

    Args:
        data_dir (str): The root directory of the dataset.
        output_dir (str): The output directory to check for existing results.
        paired (bool): Whether to use paired or unpaired data.
        util_model_path (str): Path to utility models for preprocessing.
        check_preprocessing (bool): Whether to check and generate missing preprocessing files.
    """

    def __init__(self, data_dir: str, output_dir: str = None, paired: bool = True, 
                 util_model_path: str = "Models/Human-Toolkit", check_preprocessing: bool = True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.util_model_path = util_model_path
        self.check_preprocessing = check_preprocessing
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        self.size = (1024, 768)
        self.ref_categories = ["upper", "lower", "overall", "shoe", "bag"]
        self.ref_labels = [
            0,
            1,
            2,
            3,
            4,
        ]  # 0: upper, 1: lower, 2: overall, 3: shoe, 4: bag
        self.ref_resolution = (512, 384)
        
        # Load the data
        self.data = []
        data_jsonl = os.path.join(
            self.data_dir, "test.jsonl" if paired else "test_unpair.jsonl"
        )
        if not os.path.exists(data_jsonl):
            raise FileNotFoundError(
                f"File {data_jsonl} not found, please download from https://huggingface.co/datasets/zhengchong/DressCode-MR/tree/main and put it in {self.data_dir}."
            )

        with open(data_jsonl, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                references = {
                    cat: record[cat]
                    for cat in self.ref_categories
                    if cat in record and record[cat]
                }
                if not references:
                    continue
                
                # Check if output already exists
                if self.output_dir:
                    output_filename = os.path.basename(record["person"])
                    output_path = os.path.join(self.output_dir, output_filename)
                    if os.path.exists(output_path):
                        continue  # Skip if already generated
                
                self.data.append(
                    {
                        "root": str(self.data_dir),
                        "person": record["person"],
                        "references": references,
                    }
                )
        
        # 在数据加载完成后进行预处理检查
        if self.check_preprocessing:
            preprocessing_checker = PreprocessingChecker(util_model_path)
            preprocessing_checker.check_all_dwpose_files(self.data, self.data_dir)

    def _load_image(
        self,
        path: Path,
        interpolation: int = Image.LANCZOS,
        to_tensor: bool = False,
        to_numpy: bool = False,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Union[Image.Image, torch.Tensor, np.ndarray]:
        img = Image.open(path)
        if width is not None and height is not None:
            img = center_crop_max_area_by_aspect_ratio(img, width / height)
            img = img.resize((width, height), resample=interpolation)
        else:
            img = center_crop_max_area_by_aspect_ratio(img, self.size[1] / self.size[0])
            img = img.resize((self.size[1], self.size[0]), resample=interpolation)
        if to_tensor:
            img = self.transform(img)
        if to_numpy:
            img = np.array(img)
        return img

    def _generate_person_mask(
        self,
        lip_img: np.ndarray,
        atr_img: np.ndarray,
        densepose_img: np.ndarray,
        mask_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generates a cloth-agnostic person mask from various segmentation maps.

        Args:
            lip_img (np.ndarray): LIP (Look Into Person) segmentation map.
            atr_img (np.ndarray): ATR (Active Template Regression) parsing map.
            densepose_img (np.ndarray): DensePose segmentation map.
            mask_type (Optional[str]): If specified, the part to mask (e.g., 'upper_body').
                                       If None, a general multi-reference mask is created.

        Returns:
            torch.Tensor: The generated person mask as a tensor of shape (1, H, W).
        """
        if mask_type is None:
            # Create a general mask that is agnostic to all clothing items.
            person_mask_np = multi_ref_cloth_agnostic_mask(
                densepose_img,
                lip_img,
                atr_img,
                square_cloth_mask=False,
                horizon_expand=False,
            )
        else:
            # Create a mask for a specific clothing part.
            person_mask_np = cloth_agnostic_mask(
                densepose_img, lip_img, atr_img, part=mask_type
            )

        # Convert the numpy array mask (H, W) to a tensor (1, H, W) with values in [0, 1].
        return F.to_tensor(person_mask_np)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        root = Path(sample["root"])

        # --- 1. Load Person Image ---
        person_path = root / sample["person"]
        person_img_pil = self._load_image(person_path)
        person_img = self.transform(person_img_pil)

        # --- 2. Load Pose Image ---
        dwpose_file = (
            sample["person"].replace("person", "annotations/dwpose").rsplit(".", 1)[0]
            + ".png"
        )
        dwpose_path = root / dwpose_file
        dwpose_img_pil = self._load_image(dwpose_path)
        dwpose_img = self.transform(dwpose_img_pil)
        dwpose_img = dwpose_img * 0.5 + 0.5

        # --- 3. Process Reference Images and Metadata ---
        ref_images, ref_attention_masks, ref_labels = [], [], []

        for category in self.ref_categories:
            if category in sample["references"]:
                cloth_path = root / sample["references"][category]
                cloth_img_pil = self._load_image(
                    cloth_path,
                    width=self.ref_resolution[1],
                    height=self.ref_resolution[0],
                )
                cloth_img = self.transform(cloth_img_pil)
                ref_images.append(cloth_img.clone())
                ref_attention_masks.append(1)
                ref_labels.append(self.ref_labels[self.ref_categories.index(category)])
            else:
                placeholder_img = torch.zeros(
                    3, self.ref_resolution[0], self.ref_resolution[1]
                )
                ref_images.append(placeholder_img.clone())
                ref_attention_masks.append(0)
                ref_labels.append(self.ref_labels[self.ref_categories.index(category)])

        # --- 4. Generate Person Mask ---
        def load_annotation_map(subdir: str) -> np.ndarray:
            ann_filename = (
                sample["person"]
                .replace("person", f"annotations/{subdir}")
                .rsplit(".", 1)[0]
                + ".png"
            )
            ann_path = root / ann_filename
            if ann_path.exists():
                img_pil = self._load_image(
                    ann_path, width=self.size[1], height=self.size[0]
                )
                return np.array(img_pil)
            return np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)

        lip_map = load_annotation_map("lip")
        atr_map = load_annotation_map("atr")
        densepose_map = load_annotation_map("densepose")
        person_mask = self._generate_person_mask(lip_map, atr_map, densepose_map)

        # --- 5. Return the Sample ---
        return {
            "file_names": os.path.basename(sample["person"]),
            "pixel_values": person_img,
            "masks": person_mask,
            "poses": dwpose_img,
            "ref_images": ref_images,  # List
            "ref_attention_masks": ref_attention_masks,  # List
            "ref_labels": ref_labels,  # List
        }


class DressCodeDataset(DressCodeMRDataset):
    def __init__(self, data_dir: str, output_dir: str = None, paired: bool = True, 
                 util_model_path: str = "Models/Human-Toolkit", check_preprocessing: bool = True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.util_model_path = util_model_path
        self.check_preprocessing = check_preprocessing
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        self.size = (1024, 768)
        self.ref_resolution = (1024, 768)
        self.ref_labels = {"upper": 0, "lower": 1, "overall": 2}
        
        # Load the data
        self.data = []
        data_txt = os.path.join(self.data_dir, "test_pairs_unpaired.txt")
        if not os.path.exists(data_txt):
            raise FileNotFoundError(f"File {data_txt} not found.")

        with open(data_txt, "r") as f:
            for line in f:
                # 1048404_0.png 048404_1.png upper
                person, cloth, category = line.strip().split(" ")
                if paired:
                    cloth = person.replace("0.jpg", "1.jpg")
                if category == "dresses":
                    category = "overall"
                
                # Check if output already exists
                if self.output_dir:
                    output_filename = os.path.basename(person)
                    output_path = os.path.join(self.output_dir, output_filename)
                    if os.path.exists(output_path):
                        continue  # Skip if already generated
                
                self.data.append(
                    {
                        "root": str(self.data_dir),
                        "person": os.path.join("person", person),
                        "cloth": os.path.join("cloth", cloth),
                        "category": self.ref_labels[category],
                    }
                )
        
        # 在数据加载完成后进行预处理检查
        if self.check_preprocessing:
            preprocessing_checker = PreprocessingChecker(util_model_path)
            preprocessing_checker.check_all_dwpose_files(self.data, self.data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        root = Path(sample["root"])

        # --- 1. Load Person Image ---
        person_path = root / sample["person"]
        person_img_pil = self._load_image(person_path)
        person_img = self.transform(person_img_pil)

        # --- 2. Load Cloth Image ---
        cloth_path = root / sample["cloth"]
        cloth_img_pil = self._load_image(cloth_path)
        cloth_img = self.transform(cloth_img_pil)

        # --- 3. Load Pose Image ---
        openpose_file = (
            sample["person"].replace("person", "dwpose").rsplit(".", 1)[0] + ".png"
        )
        openpose_path = root / openpose_file
        openpose_img_pil = self._load_image(openpose_path)
        openpose_img = self.transform(openpose_img_pil)
        openpose_img = openpose_img * 0.5 + 0.5

        # --- 4. Load Mask ---
        mask_path = os.path.join(
            root, sample["person"].replace("person", "mask").rsplit(".", 1)[0] + ".png"
        )
        mask_img_pil = self._load_image(mask_path)
        mask_img = self.transform(mask_img_pil)
        mask_img = mask_img * 0.5 + 0.5

        # --- 5. Return the Sample ---
        return {
            "file_names": os.path.basename(sample["person"]),
            "pixel_values": person_img,
            "masks": mask_img,
            "poses": openpose_img,
            "ref_images": [cloth_img],
            "ref_attention_masks": [1],
            "ref_labels": [sample["category"]],
        }


class VitonHDDataset(DressCodeMRDataset):
    def __init__(self, data_dir: str, output_dir: str = None, paired: bool = True, 
                 util_model_path: str = "Models/Human-Toolkit", check_preprocessing: bool = True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.util_model_path = util_model_path
        self.check_preprocessing = check_preprocessing
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )
        self.size = (1024, 768)
        self.ref_resolution = (1024, 768)
        
        # Load the data
        self.data = []
        data_txt = os.path.join(
            self.data_dir, "test_pairs.txt" if paired else "test_unpairs.txt"
        )
        if not os.path.exists(data_txt):
            raise FileNotFoundError(f"File {data_txt} not found.")

        with open(data_txt, "r") as f:
            for line in f:
                # 12544_00.jpg 14193_00.jpg
                person, cloth = line.strip().split(" ")
                
                # Check if output already exists
                if self.output_dir:
                    output_filename = os.path.basename(person)
                    output_path = os.path.join(self.output_dir, output_filename)
                    if os.path.exists(output_path):
                        continue  # Skip if already generated
                
                self.data.append(
                    {
                        "root": str(self.data_dir),
                        "person": os.path.join("test", "image", person),
                        "cloth": os.path.join("test", "cloth", cloth),
                    }
                )
        
        # 在数据加载完成后进行预处理检查
        if self.check_preprocessing:
            preprocessing_checker = PreprocessingChecker(util_model_path)
            preprocessing_checker.check_all_dwpose_files(self.data, self.data_dir)

    def __getitem__(self, idx):
        sample = self.data[idx]
        root = Path(sample["root"])

        # --- 1. Load Person Image ---
        person_path = root / sample["person"]
        person_img_pil = self._load_image(person_path)
        person_img = self.transform(person_img_pil)

        # --- 2. Load Cloth Image ---
        cloth_path = root / sample["cloth"]
        cloth_img_pil = self._load_image(cloth_path)
        cloth_img = self.transform(cloth_img_pil)

        # --- 3. Load Pose Image ---
        openpose_file = (
            sample["person"].replace("image", "dwpose").rsplit(".", 1)[0] + ".png"
        )
        openpose_path = root / openpose_file
        openpose_img_pil = self._load_image(openpose_path)
        openpose_img = self.transform(openpose_img_pil)
        openpose_img = openpose_img * 0.5 + 0.5

        # --- 4. Load Mask ---
        mask_path = os.path.join(
            root,
            sample["person"].replace("image", "agnostic-mask-catvton").rsplit(".", 1)[0]
            + ".png",
        )
        mask_img_pil = self._load_image(mask_path)
        mask_img = self.transform(mask_img_pil)
        mask_img = mask_img * 0.5 + 0.5

        # --- 5. Return the Sample ---
        return {
            "file_names": os.path.basename(sample["person"]),
            "pixel_values": person_img,
            "masks": mask_img,
            "poses": openpose_img,
            "ref_images": [cloth_img],
            "ref_attention_masks": [1],
            "ref_labels": [0],
        }


# --- Inference ---


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["dresscode-mr", "dresscode", "viton-hd"],
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    parser.add_argument("--show_skipped", action="store_true", help="Show information about skipped images")
    parser.add_argument("--util_model_path", type=str, default="Models/Human-Toolkit", help="Path to utility models for preprocessing")
    parser.add_argument("--check_preprocessing", action="store_true", default=True, help="Check and generate missing preprocessing files")
    parser.add_argument("--no_check_preprocessing", action="store_true", help="Disable preprocessing check")
    return parser.parse_args()


def count_existing_outputs(output_dir: str) -> int:
    """Count the number of existing output files in the output directory."""
    if not os.path.exists(output_dir):
        return 0
    
    count = 0
    for file in os.listdir(output_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            count += 1
    return count


def get_existing_outputs(output_dir: str) -> list:
    """Get the list of existing output filenames in the output directory."""
    if not os.path.exists(output_dir):
        return []
    
    existing_files = []
    for file in os.listdir(output_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            existing_files.append(file)
    return sorted(existing_files)


def main():
    args = parse_args()

    # --- Prepare the Dataset and Pipeline ---
    args.output_dir = os.path.join(
        args.output_dir, args.dataset, "paired" if args.paired else "unpaired"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Count existing outputs
    existing_count = count_existing_outputs(args.output_dir)
    
    print(f"Output directory: {args.output_dir}")
    print(f"Existing outputs: {existing_count} images")
    
    # 处理预处理检查参数
    check_preprocessing = args.check_preprocessing and not args.no_check_preprocessing
    if check_preprocessing:
        print(f"Preprocessing check enabled. Utility models path: {args.util_model_path}")
    else:
        print("Preprocessing check disabled.")
    
    if args.dataset == "dresscode-mr":
        dataset = DressCodeMRDataset(
            args.data_dir, 
            output_dir=args.output_dir, 
            paired=args.paired,
            util_model_path=args.util_model_path,
            check_preprocessing=check_preprocessing
        )
        pipeline = FastFitPipeline(
            base_model_path="zhengchong/FastFit-MR-1024",
            mixed_precision=args.mixed_precision,
            allow_tf32=True,
        )
    elif args.dataset == "dresscode":
        dataset = DressCodeDataset(
            args.data_dir, 
            output_dir=args.output_dir, 
            paired=args.paired,
            util_model_path=args.util_model_path,
            check_preprocessing=check_preprocessing
        )
        pipeline = FastFitPipeline(
            base_model_path="zhengchong/FastFit-SR-1024",
            mixed_precision=args.mixed_precision,
            allow_tf32=True,
        )
    elif args.dataset == "viton-hd":
        dataset = VitonHDDataset(
            args.data_dir, 
            output_dir=args.output_dir, 
            paired=args.paired,
            util_model_path=args.util_model_path,
            check_preprocessing=check_preprocessing
        )
        pipeline = FastFitPipeline(
            base_model_path="zhengchong/FastFit-SR-1024",
            mixed_precision=args.mixed_precision,
            allow_tf32=True,
        )
    else:
        raise ValueError(
            f"Invalid dataset: {args.dataset}, for now only support `dresscode-mr`"
        )
    
    print(f"Dataset loaded with {len(dataset)} samples to process")
    if args.show_skipped:
        print(f"Skipped {existing_count} already generated images")
        if existing_count > 0:
            existing_files = get_existing_outputs(args.output_dir)
            print("Skipped images:")
            for i, filename in enumerate(existing_files[:10]):  # Show first 10
                print(f"  {filename}")
            if existing_count > 10:
                print(f"  ... and {existing_count - 10} more")
    if len(dataset) == 0:
        print("All images have already been generated. Exiting.")
        return

    # --- Inference ---
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    processed_count = 0
    skipped_count = 0
    
    print(f"Starting inference with {len(dataset)} samples...")
    for sample in tqdm(dataloader, desc="Processing images"):
        try:
            image = pipeline(
                person=sample["pixel_values"],
                mask=sample["masks"],
                ref_images=sample["ref_images"],
                ref_labels=sample["ref_labels"],
                ref_attention_masks=sample["ref_attention_masks"],
                pose=sample["poses"],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=pipeline.device),
                cross_attention_kwargs=None,
            )

            # --- Save the Result ---
            for i, image in enumerate(image):
                output_path = os.path.join(args.output_dir, f"{sample['file_names'][i]}")
                image.save(output_path)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing {sample['file_names']}: {e}")
            skipped_count += 1
            continue

if __name__ == "__main__":
    main()
