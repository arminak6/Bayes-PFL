import os
import cv2
import csv
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.VPB import Context_Prompting
from models.VPB import TextEncoder
from models.model_CLIP import Load_CLIP, tokenize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    import open_clip_local as open_clip
except Exception:
    open_clip = None
try:
    from datasets import Makedataset
    DATASET_IMPORT_ERROR = None
except Exception as dataset_import_error:
    Makedataset = None
    DATASET_IMPORT_ERROR = dataset_import_error
try:
    from models.metric_and_visualization import calcuate_metric_pixel, calcuate_metric_image
    METRIC_IMPORT_ERROR = None
except Exception as metric_import_error:
    calcuate_metric_pixel = None
    calcuate_metric_image = None
    METRIC_IMPORT_ERROR = metric_import_error
try:
    from sklearn.metrics import precision_recall_curve
except Exception:
    precision_recall_curve = None
def _convert_image_to_rgb(image):
    return image.convert("RGB")
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _transform_test(n_px):
    return Compose([
        Resize((n_px,n_px), interpolation=BICUBIC),
        CenterCrop((n_px,n_px)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_existing_path(path, fallback_paths):
    if path and os.path.exists(path):
        return path
    for fallback in fallback_paths:
        if os.path.exists(fallback):
            return fallback
    return path


def _normalize_path(path):
    return str(path).replace("\\", "/")


def _safe_label(value):
    try:
        return int(value)
    except Exception:
        return 0


def _safe_score(value):
    try:
        return float(np.squeeze(value))
    except Exception:
        return 0.0


def _read_custom_meta(meta_path, mode="test"):
    with open(meta_path, "r") as f:
        raw_meta = json.load(f)

    phase_info = {}
    if isinstance(raw_meta, dict) and mode in raw_meta:
        section = raw_meta[mode]
        if isinstance(section, dict):
            phase_info = section
        elif isinstance(section, list):
            phase_info = {"object": section}
    elif isinstance(raw_meta, dict):
        for cls_name, value in raw_meta.items():
            if isinstance(value, dict) and mode in value and isinstance(value[mode], list):
                phase_info[cls_name] = value[mode]
            elif isinstance(value, list) and mode == "test":
                phase_info[cls_name] = value

    if not phase_info:
        raise ValueError(f"Unsupported meta format in {meta_path}")
    return phase_info


def _scan_custom_test_dir(data_path):
    test_root = os.path.join(data_path, "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(
            f"Custom dataset mode expects '{test_root}' or a valid --meta_path."
        )

    valid_exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    phase_info = {"object": []}
    for specie_dir in sorted(os.listdir(test_root)):
        specie_path = os.path.join(test_root, specie_dir)
        if not os.path.isdir(specie_path):
            continue
        is_good = specie_dir.lower() in {"good", "normal", "ok"}
        for file_name in sorted(os.listdir(specie_path)):
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in valid_exts:
                continue
            rel_path = _normalize_path(os.path.join("test", specie_dir, file_name))
            phase_info["object"].append(
                {
                    "img_path": rel_path,
                    "mask_path": "",
                    "anomaly": 0 if is_good else 1,
                    "specie_name": "good" if is_good else "anomaly",
                }
            )

    if not phase_info["object"]:
        raise RuntimeError(f"No images found under {test_root}")
    return phase_info


class CustomInferenceDataset(Dataset):
    def __init__(self, root, transform, target_transform, mode="test", meta_path=""):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.data_all = []
        self.cls_names = []

        resolved_meta = meta_path if meta_path and os.path.exists(meta_path) else os.path.join(root, "meta.json")
        if os.path.exists(resolved_meta):
            meta_info = _read_custom_meta(resolved_meta, mode=mode)
        else:
            meta_info = _scan_custom_test_dir(root)

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            for raw_item in meta_info[cls_name]:
                if "img_path" not in raw_item:
                    continue
                anomaly = _safe_label(raw_item.get("anomaly", 0))
                self.data_all.append(
                    {
                        "img_path": raw_item["img_path"],
                        "mask_path": raw_item.get("mask_path", ""),
                        "cls_name": raw_item.get("cls_name", cls_name),
                        "specie_name": raw_item.get("specie_name", "good" if anomaly == 0 else "anomaly"),
                        "anomaly": anomaly,
                    }
                )

        if not self.data_all:
            raise RuntimeError("CustomInferenceDataset found no valid samples.")

    def __len__(self):
        return len(self.data_all)

    def get_cls_names(self):
        return self.cls_names

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return _normalize_path(path)
        return _normalize_path(os.path.join(self.root, path))

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = self._resolve_path(data["img_path"])
        mask_path = self._resolve_path(data["mask_path"]) if data.get("mask_path") else ""
        anomaly = _safe_label(data.get("anomaly", 0))

        img = Image.open(img_path)
        if anomaly == 1 and mask_path and os.path.exists(mask_path):
            img_mask = np.array(Image.open(mask_path).convert("L")) > 0
            img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode="L")
        else:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8), mode="L")

        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(img_mask) if self.target_transform is not None else img_mask

        return {
            "img": img,
            "img_mask": img_mask,
            "cls_name": data["cls_name"],
            "anomaly": anomaly,
            "img_path": img_path,
        }


def _compute_best_f1_threshold(results):
    if precision_recall_curve is None:
        return None, None
    labels = np.array([_safe_label(v) for v in results["gt_sp"]], dtype=np.int32)
    scores = np.array([_safe_score(v) for v in results["pr_sp"]], dtype=np.float32)
    unique_labels = sorted(np.unique(labels).tolist())
    if unique_labels != [0, 1]:
        return None, None
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return None, None
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def _save_prediction_csv(results, save_path, threshold=None, threshold_source="none"):
    csv_path = os.path.join(save_path, "predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "image_path",
                "class_name",
                "gt_label",
                "pred_score",
                "pred_label",
                "threshold",
                "threshold_source",
            ]
        )
        for idx, img_path in enumerate(results["path"]):
            score = _safe_score(results["pr_sp"][idx])
            pred_label = int(score >= threshold) if threshold is not None else ""
            writer.writerow(
                [
                    idx,
                    _normalize_path(img_path),
                    results["cls_names"][idx],
                    _safe_label(results["gt_sp"][idx]),
                    f"{score:.6f}",
                    pred_label,
                    "" if threshold is None else f"{threshold:.6f}",
                    threshold_source,
                ]
            )
    return csv_path


def _save_overlay_images(results, save_path, image_size):
    vis_root = os.path.join(save_path, "custom_vis")
    os.makedirs(vis_root, exist_ok=True)

    for idx, img_path in enumerate(results["path"]):
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
        raw_img = cv2.resize(raw_img, (image_size, image_size))
        anomaly_map = np.squeeze(results["anomaly_maps"][idx]).astype(np.float32)
        map_min = float(np.min(anomaly_map))
        map_max = float(np.max(anomaly_map))
        if map_max > map_min:
            anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
        else:
            anomaly_map = np.zeros_like(anomaly_map)

        heat_map = cv2.applyColorMap((anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.65, heat_map, 0.35, 0)
        score = _safe_score(results["pr_sp"][idx])
        label = _safe_label(results["gt_sp"][idx])
        name = os.path.splitext(os.path.basename(img_path))[0]
        save_name = f"{idx:04d}_{name}_label{label}_score{score:.4f}.png"
        stacked = np.concatenate([raw_img, overlay], axis=1)
        cv2.imwrite(os.path.join(vis_root, save_name), stacked)


def _has_pixel_gt(results):
    for idx, gt_label in enumerate(results["gt_sp"]):
        if _safe_label(gt_label) != 1:
            continue
        mask = results["imgs_masks"][idx]
        if torch.is_tensor(mask):
            if torch.any(mask > 0.5).item():
                return True
        else:
            if np.any(np.asarray(mask) > 0.5):
                return True
    return False


def _has_both_labels(results):
    labels = {_safe_label(item) for item in results["gt_sp"]}
    return labels == {0, 1}


def test(args):
    image_size = args.image_size
    save_path = args.save_path
    dataset_name = args.dataset

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
    txt_path = os.path.join(save_path, 'log.txt')

    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    args.vision_width = model_configs["vision_cfg"]['width']
    args.text_width = model_configs['text_cfg']['width']
    args.embed_dim = model_configs['embed_dim']

    args.pretrained_path = _resolve_existing_path(args.pretrained_path, ["./ViT-L-14-336px.pt"])
    args.checkpoint_path = _resolve_existing_path(args.checkpoint_path, ["./train_visa.pth", "./train_mvtec.pth"])
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(f"CLIP weight not found: {args.pretrained_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Bayes-PFL checkpoint not found: {args.checkpoint_path}")

    # We retained the OpenCLIP interface to enable Bayes-PFL to support a broader range of backbones.
    # -------------------------------------------------------------------------------------------------
    
    # Example 1 : The pretrained model from huggingface laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
    '''
    model_clip, _, _ = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", img_size= image_size) 
    tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
    model_clip = model_clip.to(device)
    model_clip.eval()
    '''

    # Example 2 : The pretrained model from OpenAI CLIP
    '''
    model_clip, _, _ = open_clip.create_model_and_transforms(args.model, pretrained= args.pretrained, img_size= image_size) 
    tokenizer = open_clip.get_tokenizer(args.model)
    model_clip = model_clip.to(device)
    model_clip.eval()
    '''
    
    # -------------------------------------------------------------------------------------------------
    
    # This is from our own implementation of the CLIP model, which only supports the OpenAI pretrained models ViT-B-16, ViT-L-14, and ViT-L-14-336.
    model_clip , _ , _ = Load_CLIP(image_size, args.pretrained_path , device=device) 
    tokenizer = tokenize
    model_clip.to(device)
    model_clip.eval()

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    PFL_TextEncoder = TextEncoder(model_clip, args)
    MyModel = Context_Prompting(args = args).to(device)
    MyModel.eval()

    checkpoint = torch.load(args.checkpoint_path, map_location= device)
    MyModel.load_state_dict(checkpoint["MyModel"], strict= True)

    preprocess_test = _transform_test(image_size)


    if dataset_name.lower() == "custom":
        target_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        custom_dataset = CustomInferenceDataset(
            root=args.data_path,
            transform=preprocess_test,
            target_transform=target_transform,
            mode="test",
            meta_path=args.meta_path,
        )
        test_dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)
        obj_list = custom_dataset.get_cls_names()
    else:
        if Makedataset is None:
            raise ImportError(f"Failed to import standard dataset loader: {DATASET_IMPORT_ERROR}")
        Make_dataset_test = Makedataset(
            train_data_path=args.data_path,
            preprocess_test=preprocess_test,
            mode="test",
            image_size=args.image_size,
        )
        test_dataloader, obj_list = Make_dataset_test.mask_dataset(
            name=dataset_name, product_list=None, batchsize=1, shuf=False
        )

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []

    results['pr_sp'] = []
    results['gt_sp'] = []
    results['path'] = []

    id = 0
    if "epoch_post" in args.checkpoint_path:
        stage = int(args.checkpoint_path[:-4].split("_")[-1])
    else:
        stage = 2
    for items in tqdm(test_dataloader):
        id = id + 1
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        results['gt_sp'].append(items['anomaly'].item())
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        
        with torch.no_grad():
            image_features, _ , patch_tokens =  model_clip.encode_image(image, args.features_list)
            text_embeddings, _ = MyModel.forward_ensemble(PFL_TextEncoder, image_features, patch_tokens ,cls_name, device, tokenizer, mode = "test") # # B * R text embeddings
            temp_cls = 0
            pro_img, anomaly_maps_list = MyModel(text_embeddings, image_features, patch_tokens, stage = stage, mode = "test")
            pro_img = pro_img.squeeze(2)
            for i in range(args.prompt_num * args.sample_num): # B * R anomaly scores
                text_probs = torch.cat([pro_img[:,i].unsqueeze(0), pro_img[:,i + args.prompt_num * args.sample_num].unsqueeze(0)], dim = 1).softmax(dim = -1)
                temp_cls = temp_cls + text_probs[0, 1]
            temp_cls = temp_cls / (args.prompt_num * args.sample_num)

            anomaly_maps = []
            for num in range(len(anomaly_maps_list)):
                anomaly_map = anomaly_maps_list[num]
                for i in range(args.prompt_num * args.sample_num):  # B * R anomaly maps
                    temp = torch.softmax(torch.stack([anomaly_map[:,i,:,:], anomaly_map[:,i+(args.prompt_num * args.sample_num) ,:,:]], dim = 1), dim =1)
                    anomaly_maps.append(temp[:, 1, :, :].cpu().numpy())
            anomaly_map = np.mean(anomaly_maps, axis=0)[0]
            results['anomaly_maps'].append(anomaly_map)
            results['pr_sp'].append(temp_cls.cpu().numpy())
        path = items['img_path']
        path = [_normalize_path(p) for p in path]
        results['path'].extend(path)

    cls_threshold = None
    threshold_source = "none"
    best_f1 = None
    if args.cls_threshold is not None:
        cls_threshold = float(args.cls_threshold)
        threshold_source = "manual"
    elif args.auto_cls_threshold:
        cls_threshold, best_f1 = _compute_best_f1_threshold(results)
        if cls_threshold is not None:
            threshold_source = "best_f1_on_current_set"

    csv_path = _save_prediction_csv(results, save_path, threshold=cls_threshold, threshold_source=threshold_source)
    _save_overlay_images(results, save_path, image_size)
    logger.info(f"Saved prediction CSV to: {csv_path}")
    logger.info(f"Saved overlay visualizations to: {os.path.join(save_path, 'custom_vis')}")
    if cls_threshold is not None:
        if best_f1 is None:
            logger.info(f"Classification threshold used: {cls_threshold:.6f} (source: {threshold_source})")
        else:
            logger.info(
                f"Classification threshold used: {cls_threshold:.6f} "
                f"(source: {threshold_source}, best_f1={best_f1:.4f})"
            )
    else:
        logger.info("No classification threshold applied. Set --cls_threshold or --auto_cls_threshold.")

    datasets_only_classification =  ["HeadCT", "BrainMRI", "Br35H", "custom"]  # These datasets lack ground truth, so only zero-shot anomaly classification metrics are calculated.
    metric_mode = args.metric_mode.lower()
    has_pixel_gt = _has_pixel_gt(results)
    has_binary_label = _has_both_labels(results)

    if metric_mode == "auto":
        if args.dataset in datasets_only_classification:
            metric_mode = "image"
        elif has_pixel_gt:
            metric_mode = "pixel"
        elif has_binary_label:
            metric_mode = "image"
        else:
            metric_mode = "none"

    if metric_mode == "pixel" and not has_pixel_gt:
        logger.warning("metric_mode=pixel requested but no valid pixel masks found. Falling back to image-level metrics.")
        metric_mode = "image" if has_binary_label else "none"

    if metric_mode == "image" and not has_binary_label:
        logger.warning("Image-level metrics require both normal and anomaly labels. Skipping metric computation.")
        metric_mode = "none"

    if metric_mode != "none" and (calcuate_metric_pixel is None or calcuate_metric_image is None):
        logger.warning(f"Metric utilities unavailable ({METRIC_IMPORT_ERROR}). Skipping metric computation.")
        metric_mode = "none"

    if metric_mode == "pixel":
        calcuate_metric_pixel(results, obj_list, logger, alpha = 0.5 , sigm = 8, args = args)
    elif metric_mode == "image":
        calcuate_metric_image(results, obj_list, logger, alpha = 0.5 , sigm = 8, args = args)
    else:
        logger.info("Metric computation skipped (metric_mode=none).")

import shutil
def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
if __name__ == '__main__':

    # Industry_Datasets = ["mvtec","visa", "BTAD", "RSDD", "KSDD2", "DAGM", "DTD"]
    # Medical_Datasets = ["HeadCT", "BrainMRI", "Br35H", "ISIC", "ClinicDB", "ColonDB", "Kvasir", "Endo"]

    # datasets_no_good = ["ISIC", "ClinicDB", "ColonDB", "Kvasir", "Endo"] # These datasets do not contain normal samples, so only zero-shot anomaly segmentation metrics are calculated.
    # datasets_only_classification = ["HeadCT", "BrainMRI", "Br35H"]  # These datasets lack ground truth, so only zero-shot anomaly classification metrics are calculated.

    parser = argparse.ArgumentParser("Bayes-PFL", add_help=True)

    # Model
    parser.add_argument("--dataset", type=str, default='mvtec', help="Testing dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--image_size", type=int, default= 518, help="image size")
    parser.add_argument("--pretrained", type=str, default="openai", help="Source of pretrained weight")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # path
    parser.add_argument("--data_path", type=str, default="./dataset/mvisa/data", help="Testing dataset path")
    parser.add_argument("--save_path", type=str, default='./results/test_mvtec', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip_local/model_configs/ViT-L-14-336.json', help="model configs")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt", help="Original pretrained CLIP path")
    parser.add_argument("--checkpoint_path", type=str, default="./bayes_weight/train_visa.pth", help='path to checkpoint')
    parser.add_argument("--meta_path", type=str, default="", help="Optional meta JSON path for custom dataset mode")
    parser.add_argument("--metric_mode", type=str, default="auto", choices=["auto", "pixel", "image", "none"],
                        help="Metric mode: auto/pixel/image/none")
    parser.add_argument(
        "--cls_threshold",
        type=float,
        default=None,
        help="Manual threshold for score->label classification (pred_label = score >= threshold).",
    )
    parser.add_argument(
        "--auto_cls_threshold",
        action="store_true",
        help="If set and labels exist, choose threshold by best F1 on the current evaluated set.",
    )

    # hyper-parameter
    parser.add_argument('-nf', '--num_flows', type=int, default=10,
                        metavar='NUM_FLOWS', help='Flow length')  # $K$ in the main text
    parser.add_argument("--prompt_context_len", type=int, default=5, help="The length of learnable context vectors")  # $P$ in the main text
    parser.add_argument("--prompt_num", type=int, default=3, help="The number of prompts in the prompt bank") # $B$ in the main text
    parser.add_argument("--prompt_state_len", type=int, default=5, help="The length of learnable state vectors")  # $Q$ in the main text
    parser.add_argument("--sample_num", type= int, default= 10, help="The number of Monte Carlo sampling interations")   # $R$ in the main text


    parser.add_argument("--device_id", type=int, default= 2, help="GPU ID")
    parser.add_argument("--seed", type=int, default= 111, help="save frequency")


    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)

    if "ceshi" in args.save_path:
        move(args.save_path)
    setup_seed(args.seed)
    test(args)

