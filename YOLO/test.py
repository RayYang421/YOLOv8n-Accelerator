from ultralytics import YOLO


if __name__ == '__main__':
    # 載入模型
    model = YOLO('/home/master_114/q36141252/Himax/runs/Baseline_IR_20260403/IR_Branch_n_0402/weights/best.pt')  
    # 驗證模型
    metrics = model.val(
        data='/home/master_114/q36141252/ultralytics/ultralytics/cfg/datasets/LLVIP_IR.yaml',
        split='val',  # （str）用於驗證的資料集拆分，例如 'val'、'test' 或 'train'
        batch=4,  # （int）每批次的圖像數量（-1 表示自動批次處理）
        imgsz=640,  # 輸入圖像的尺寸，可以是整數或 w, h
        rect=False,
        device=1,  # 執行的裝置，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=4,  # 資料加載的工作執行緒數量（每個 DDP 程序）
        save_json=False,  # 是否將結果儲存為 JSON 檔案
        save_hybrid=False,  
        conf=0.001,  # 偵測目標的置信度閾值（預設為 0.25 用於預測，0.001 用於驗證）
        iou=0.45,  
        project='runs/val',  # 專案名稱（可選）
        name='2026' \
        '',  # 實驗名稱，結果儲存在 'project/name' 目錄下（可選）
        max_det=300,  
        half=False,  # 是否使用半精度（FP16）
        dnn=False,  # 是否使用 OpenCV DNN 進行 ONNX 推論
        plots=True,  # 是否在訓練／驗證期間儲存圖像
        verbose=False,
    )


    m = model.model
    n_params = sum(p.numel() for p in m.parameters())
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("params:", n_params)
    print("trainable:", n_trainable)


