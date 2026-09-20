from ultralytics import YOLO

model = YOLO('yolo26m.pt')

model.train(data="/mnt/workspace/ty/xray/datasets/yolo/data.yaml", 
            epochs=500, 
            imgsz=640, 
            batch=32,
            device=[0, 1],

            cache=True,
            name='raw-attempt-1',

            seed=1234,

            # Augment nhe cho VinDr-CXR: giu cau truc giai phau, tranh bien doi mau qua manh.
            degrees=5.0,
            translate=0.05,
            scale=0.15,
            shear=2.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,

            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.10,

            mosaic=0.3,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=20,
            
            )
