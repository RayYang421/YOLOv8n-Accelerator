from ultralytics import YOLO

def main():
    model = YOLO("/home/master_114/q36141252/ultralytics/ultralytics/cfg/models/v8/yolov8n.yaml")
    # print("scale =", model.model.yaml.get("scale"))
    print("yaml =", model.model.yaml.get("yaml_file", None)) 
    model.train(
        data="/home/master_114/q36141252/ultralytics/ultralytics/cfg/datasets/LLVIP_IR.yaml",
        epochs=100,
        imgsz=512,
        batch=4,
        device=1,
        project="/home/master_114/q36141252/Himax/runs/Baseline_IR_20260403",
        name="IR_Branch_n_0402",
        seed=42,
        deterministic=True,
        workers=8,
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0, 
        hsv_s=0, 
        hsv_v=0,
        flipud=0.0,
        fliplr=0.0,
        close_mosaic=0,
        val=False,
        plots=True,   
    )

if __name__ == "__main__":
    main()
