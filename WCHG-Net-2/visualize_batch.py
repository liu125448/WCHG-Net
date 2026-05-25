import os
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from collections import Counter


import model_hgnn as model_my
try:
    import model_baseline as model_base
except ImportError:
    model_base = None
from VisualizeH.gradcam.GradCAM import GradCAM, show_cam_on_image


# -------- 1) 让 GradCAM 的输入=单张图，输出=2D 特征张量 --------
class ReIDFeatWrapper(nn.Module):
    def __init__(self, net: nn.Module, modal: int = 1, which: str = "feat"):
        super().__init__()
        self.net = net
        self.modal = modal  # 1=visible, 2=thermal
        self.which = which  # "feat" 或 "x_pool"

    def forward(self, x):
        # eval 下返回 (l2norm(x_pool), l2norm(feat))
        x_pool, feat = self.net(x, x, x, x, self.modal)
        return feat if self.which == "feat" else x_pool


def preprocess(img_path, h=384, w=192):
    img = Image.open(img_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    x = tfm(img).unsqueeze(0)

    # show_cam_on_image 用的 0~1 RGB
    img_vis = img.resize((w, h))
    img_vis = np.array(img_vis).astype(np.float32) / 255.0
    return img_vis, x


def save_vis(vis_np_uint8, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(vis_np_uint8).save(save_path)


def is_image_file(fn: str):
    fn_l = fn.lower()
    return fn_l.endswith((".jpg", ".jpeg", ".png", ".bmp"))


def list_sysu_images(sysu_root: str):
    """
    严格按：
      SYSU/camX/ID/*.jpg
    camX 下只包含 ID 文件夹；ID 下包含多张图片。
    """
    sysu_root = os.path.abspath(sysu_root)

    cams = [d for d in os.listdir(sysu_root)
            if os.path.isdir(os.path.join(sysu_root, d)) and d.lower().startswith("cam")]
    cams = sorted(cams)

    paths = []
    for cam in cams:
        cam_dir = os.path.join(sysu_root, cam)

        # cam_dir 下只有 ID 文件夹
        ids = [d for d in os.listdir(cam_dir)
               if os.path.isdir(os.path.join(cam_dir, d))]
        ids = sorted(ids)

        for pid in ids:
            pid_dir = os.path.join(cam_dir, pid)
            # pid_dir 下是多张图片
            for fn in sorted(os.listdir(pid_dir)):
                fp = os.path.join(pid_dir, fn)
                if os.path.isfile(fp) and is_image_file(fp):
                    paths.append(fp)

    return paths


def run_one(cam_obj, img_path, save_path, h, w, device, target_category=None):
    img_vis, x = preprocess(img_path, h=h, w=w)
    x = x.to(device)

    grayscale_cam = cam_obj(input_tensor=x, target_category=target_category)[0]
    vis = show_cam_on_image(img_vis, grayscale_cam, use_rgb=True)
    save_vis(vis, save_path)


def _extract_state_dict(ckpt):
    # 1) ckpt 是整包 dict 的情况
    if isinstance(ckpt, dict):
        for k in ["state_dict", "net", "model", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], (dict, OrderedDict)):
                return ckpt[k]
    # 2) ckpt 本身就是 state_dict（OrderedDict/普通dict）
    if isinstance(ckpt, (dict, OrderedDict)):
        return ckpt
    raise ValueError("Unknown checkpoint format")

def _strip_prefix(sd, prefix="module."):
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        new_sd[k] = v
    return new_sd

def _load_state_dict(net, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    sd = _strip_prefix(sd, "module.")

    msg = net.load_state_dict(sd, strict=False)
    print("[CKPT]", ckpt_path)
    print("  loaded params:", len(sd))
    print("  missing:", len(msg.missing_keys))
    print("  unexpected:", len(msg.unexpected_keys))
    print("  missing head:", msg.missing_keys[:10])
    print("  unexpected head:", msg.unexpected_keys[:10])
    print(Counter(k.split('.')[0] for k in msg.missing_keys).most_common(20))


def build_net(embed_net_fn, num_classes: int, ckpt_path: str, device: torch.device):
    net = embed_net_fn(num_classes, no_local="on", gm_pool="on", arch="resnet50")
    _load_state_dict(net, ckpt_path)
    net = net.to(device).eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_my", type=str, default="your_wchgnet_checkpoint.t")
    ap.add_argument("--ckpt_base", type=str, default="your_baseline_checkpoint.t")
    ap.add_argument("--sysu_root", type=str, default="./dataset/SYSU-MM01", help="SYSU root directory")
    ap.add_argument("--out_root", type=str, default="./heatmap", help="输出根目录")
    ap.add_argument("--num_classes", type=int, default=395)
    ap.add_argument("--modal", type=int, default=1, help="1=visible, 2=thermal")
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--w", type=int, default=192)
    ap.add_argument("--max_imgs", type=int, default=-1)
    ap.add_argument("--gpu", type=int, default=0, help="指定GPU编号；-1 用CPU")
    args = ap.parse_args()

    # ---- device / gpu
    if args.gpu < 0 or (not torch.cuda.is_available()):
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    else:
        # 注意：如果你外面设置了 CUDA_VISIBLE_DEVICES，那么这里的 gpu 编号是“可见GPU的相对编号”
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[INFO] Using GPU cuda:{args.gpu}")

    os.makedirs(args.out_root, exist_ok=True)

    # ---- load model
    print("[INFO] Loading MY model:", args.ckpt_my)
    net_my = build_net(model_my.embed_net, args.num_classes, args.ckpt_my, device)
    wrapper_my = ReIDFeatWrapper(net_my, modal=args.modal, which="feat").to(device)

    # ---- target layers（和你最初脚本一致）
    my_out = net_my.cmgr.spectral_readout

    cam_my = GradCAM(model=wrapper_my, target_layers=[my_out])
    cam_base = None
    if model_base is not None and args.ckpt_base:
        print("[INFO] Loading BASELINE model:", args.ckpt_base)
        net_base = build_net(model_base.embed_net, args.num_classes, args.ckpt_base, device)
        wrapper_base = ReIDFeatWrapper(net_base, modal=args.modal, which="feat").to(device)
        base_out = net_base.base_resnet.base.layer3[-1].conv3
        cam_base = GradCAM(model=wrapper_base, target_layers=[base_out])

    # 如果你还想画 layer4：推荐 conv3，少条带伪影
    layer4_conv3 = net_my.base_resnet.base.layer4[-1].conv3
    cam_l4 = GradCAM(model=wrapper_my, target_layers=[layer4_conv3])

    sysu_root = os.path.abspath(args.sysu_root)
    out_root = os.path.abspath(args.out_root)

    img_paths = list_sysu_images(sysu_root)
    if args.max_imgs > 0:
        img_paths = img_paths[:args.max_imgs]

    if len(img_paths) == 0:
        print(f"[ERR] No images found under: {sysu_root}")
        return

    print(f"[INFO] Found {len(img_paths)} images under {sysu_root}")
    print(f"[INFO] Saving to {out_root} (keep cam/id structure)")

    for i, img_path in enumerate(img_paths, 1):
        # rel_path: cam1/0001/xxx.jpg
        rel_path = os.path.relpath(img_path, sysu_root)
        rel_dir = os.path.dirname(rel_path)
        base = os.path.splitext(os.path.basename(rel_path))[0]

        # 输出：OUT/cam1/0001/xxx__layer3.png 等
        out_dir = os.path.join(out_root, rel_dir)
        out_my = os.path.join(out_dir, f"{base}__my.png")
        out_base = os.path.join(out_dir, f"{base}__base.png")
        out_l4 = os.path.join(out_dir, f"{base}__layer4.png")

        try:
            run_one(cam_my, img_path, out_my, args.h, args.w, device)
            if cam_base is not None:
                run_one(cam_base, img_path, out_base, args.h, args.w, device)
            run_one(cam_l4, img_path, out_l4, args.h, args.w, device)
        except Exception as e:
            print(f"[WARN] failed: {img_path} | {e}")

        if i % 200 == 0 or i == len(img_paths):
            print(f"[OK] {i}/{len(img_paths)}")

    print("[DONE]")


if __name__ == "__main__":
    # python visualize_batch.py --gpu 0
    main()
