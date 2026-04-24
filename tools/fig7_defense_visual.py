"""Figure 7 - visualise what each defense does to the adversarial image.

For a chosen epsilon (default 0.031), render a 4-column panel:

    col 0 : x_0^clean
    col 1 : x_0^adv  (undefended)
    col 2 : purify(x_0^adv)  under each method in rows
    col 3 : |x_0^adv - purify(x_0^adv)|  x10   (what the defense removed)

This lets the reader SEE that bits:4 erases the adversarial noise while
preserving the clean image.

Inputs (all CPU, read from PNGs already on disk):
    --attackB_run_dir : e.g. src/runs/.../sample_idx4729/attackB-linf-eps_img0.031-40steps
    --out
"""
from __future__ import annotations
import argparse, io, os
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


def _load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _jpeg(img_u8, q):
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _blur(img_u8, sigma):
    pil = Image.fromarray(img_u8).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(pil)


def _bits(img_u8, bits):
    levels = 2 ** int(bits) - 1
    x = img_u8.astype(np.float32) / 255.0
    xq = np.round(x * levels) / levels
    return (np.clip(xq, 0, 1) * 255).astype(np.uint8)


def _residual(a, b, amp=10):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    d = np.clip(d * amp, 0, 255).astype(np.uint8)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attackB_run_dir", required=True,
                   help="e.g. .../sample_idx4729/attackB-linf-eps_img0.031-40steps")
    p.add_argument("--out",             required=True)
    args = p.parse_args()

    x_clean = _load(os.path.join(args.attackB_run_dir, "x0_clean.png"))
    x_adv   = _load(os.path.join(args.attackB_run_dir, "x0_adv.png"))

    plans = [("bits:4",  lambda a: _bits(a, 4)),
             ("bits:6",  lambda a: _bits(a, 6)),
             ("jpeg:75", lambda a: _jpeg(a, 75)),
             ("jpeg:50", lambda a: _jpeg(a, 50)),
             ("blur:1.5",lambda a: _blur(a, 1.5)),
             ("blur:0.5",lambda a: _blur(a, 0.5))]

    plt.rcParams.update({"font.size": 11})
    nrows = len(plans) + 1
    fig, axes = plt.subplots(nrows, 4, figsize=(11, 2.3 * nrows),
                             gridspec_kw={"hspace": 0.15, "wspace": 0.05})

    axes[0, 0].imshow(x_clean); axes[0, 0].set_title("clean $x_0$", fontsize=11)
    axes[0, 1].imshow(x_adv);   axes[0, 1].set_title("adversarial $x_0$", fontsize=11)
    axes[0, 2].imshow(_residual(x_clean, x_adv))
    axes[0, 2].set_title(r"$|x_0 - x_0^{\mathrm{adv}}|\times 10$", fontsize=11)
    axes[0, 3].axis("off")
    for a in axes[0, :3]:
        a.set_xticks([]); a.set_yticks([])

    for r, (name, fn) in enumerate(plans, start=1):
        pur = fn(x_adv)
        pur_clean = fn(x_clean)
        axes[r, 0].imshow(pur_clean); axes[r, 0].set_title(f"purify({name})(clean)", fontsize=10)
        axes[r, 1].imshow(pur);       axes[r, 1].set_title(f"purify({name})(adv)",   fontsize=10)
        axes[r, 2].imshow(_residual(x_adv, pur))
        axes[r, 2].set_title(r"$|x_0^{\mathrm{adv}} - \mathrm{purify}(x_0^{\mathrm{adv}})|\times 10$",
                             fontsize=10)
        axes[r, 3].imshow(_residual(x_clean, pur))
        axes[r, 3].set_title(r"$|x_0 - \mathrm{purify}(x_0^{\mathrm{adv}})|\times 10$",
                             fontsize=10)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])

    eps_str = os.path.basename(os.path.normpath(args.attackB_run_dir))
    fig.suptitle(f"What each D2 defense does to $x_0^{{\\mathrm{{adv}}}}$   ({eps_str})",
                 fontsize=13, y=0.995)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"[fig7] -> {args.out}")


if __name__ == "__main__":
    main()
