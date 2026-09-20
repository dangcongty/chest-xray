import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def letterbox(image, size=640):
    """
    Resize giữ nguyên tỷ lệ, sau đó padding thành size x size.
    """
    w, h = image.size

    scale = min(size / w, size / h)

    new_w = round(w * scale)
    new_h = round(h * scale)

    # Resize
    image = image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    # Tạo canvas đen 640x640
    canvas = Image.new("L", (size, size), color=0)

    # Đặt ảnh vào giữa
    x = (size - new_w) // 2
    y = (size - new_h) // 2

    canvas.paste(image, (x, y))

    return canvas

def convert_one(args):
    dicom_path, input_dir, output_dir = args

    try:
        ds = pydicom.dcmread(dicom_path)

        # Decode DICOM / JPEG2000
        img = ds.pixel_array.astype(np.float32)

        # Rescale
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        img = img * slope + intercept

        # Windowing
        center = ds.get("WindowCenter")
        width = ds.get("WindowWidth")

        # MultiValue -> lấy giá trị đầu
        if center is not None and width is not None:
            try:
                center = float(center[0])
            except (TypeError, IndexError):
                center = float(center)

            try:
                width = float(width[0])
            except (TypeError, IndexError):
                width = float(width)

            low = center - width / 2
            high = center + width / 2
        else:
            # Fallback nếu DICOM không có WindowCenter/Width
            low = float(img.min())
            high = float(img.max())

        if high <= low:
            return False, str(dicom_path), "Invalid pixel range"

        # Clip + normalize 8-bit
        img = np.clip(img, low, high)
        img = (img - low) / (high - low) * 255.0

        # MONOCHROME1 cần invert
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            img = 255.0 - img

        img = np.clip(img, 0, 255).astype(np.uint8)
        

        # Giữ cấu trúc folder
        relative_path = dicom_path.relative_to(input_dir)

        output_path = output_dir / relative_path
        output_path = output_path.with_suffix(".png")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(img)
        img = letterbox(img, 640)
        img.save(output_path)

        return True, str(dicom_path), str(output_path)

    except Exception as e:
        return False, str(dicom_path), str(e)


def main():
    input_dir = Path("/mnt/workspace/ty/xray/datasets/dicom/train")
    output_dir = Path("/mnt/workspace/ty/xray/datasets/png/images")

    # Tìm DICOM
    files = list(input_dir.rglob("*.dicom"))

    print(f"Found {len(files):,} DICOM files")

    # Có thể chỉnh số process ở đây
    workers = min(16, os.cpu_count() or 4)

    print(f"Using {workers} workers")

    tasks = [
        (path, input_dir, output_dir)
        for path in files
    ]

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(convert_one, task)
            for task in tasks
        ]

        for i, future in enumerate(as_completed(futures), 1):
            ok, src, result = future.result()

            if ok:
                success += 1
            else:
                failed += 1
                print(f"\nFAILED: {src}")
                print(f"  {result}")

            if i % 100 == 0 or i == len(files):
                print(
                    f"\r{i:,}/{len(files):,} | "
                    f"OK: {success:,} | "
                    f"Failed: {failed:,}",
                    end="",
                    flush=True
                )

    print()
    print("Done")
    print(f"Success: {success:,}")
    print(f"Failed:  {failed:,}")


if __name__ == "__main__":
    main()