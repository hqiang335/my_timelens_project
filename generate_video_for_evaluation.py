import os
import re
import cv2

# 图片所在目录
IMG_DIR = "RGB_resOut_HQEVFI/mix_Expv8_largex16_adamLPIPS/Validation_Visual_Examples/images/10"

# 输出视频路径（可以根据需要自己改）
FPS = 20
OUT_GT_MP4  = f"RGB_resOut_HQEVFI/mix_Expv8_largex16_adamLPIPS/Validation_Visual_Examples/pre_gt_{FPS}fps.mp4"
OUT_RES_MP4 = f"RGB_resOut_HQEVFI/mix_Expv8_largex16_adamLPIPS/Validation_Visual_Examples/pre_res_{FPS}fps.mp4"


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def collect_images(img_dir, suffix):
    files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg") and f.endswith(suffix)
    ]
    files = sorted(files, key=natural_key)
    return files


def make_video(img_dir, filenames, out_path, fps):
    if len(filenames) == 0:
        raise ValueError(f"在 {img_dir} 中没找到任何图片，用于输出 {out_path}")

    first_path = os.path.join(img_dir, filenames[0])
    first_img = cv2.imread(first_path)
    if first_img is None:
        raise RuntimeError(f"无法读取首帧图片: {first_path}")

    h, w = first_img.shape[:2]
    size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)

    print(f"开始合成视频: {out_path}")
    print(f"   帧率: {fps}, 分辨率: {w}x{h}, 总帧数: {len(filenames)}")

    for i, name in enumerate(filenames):
        img_path = os.path.join(img_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue

        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        writer.write(img)

        if (i + 1) % 50 == 0 or (i + 1) == len(filenames):
            print(f"   已写入帧数: {i + 1}/{len(filenames)}")

    writer.release()
    print(f"完成输出: {out_path}\n")


def main():
    gt_imgs  = collect_images(IMG_DIR,  "_gt.jpg")
    res_imgs = collect_images(IMG_DIR, "_res.jpg")

    print(f"在 {IMG_DIR} 中找到 {len(gt_imgs)} 张 GT 图片，{len(res_imgs)} 张合成帧图片。")

    if gt_imgs:
        make_video(IMG_DIR, gt_imgs, OUT_GT_MP4, FPS)
    else:
        print("没找到任何 *_gt.jpg 图片，跳过 GT 视频生成。")

    if res_imgs:
        make_video(IMG_DIR, res_imgs, OUT_RES_MP4, FPS)
    else:
        print("没找到任何 *_res.jpg 图片，跳过合成帧视频生成。")


if __name__ == "__main__":
    main()
