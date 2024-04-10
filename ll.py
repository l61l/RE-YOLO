from ultralytics import YOLO
# from ptflops import get_model_complexity_info
# import torch
# import torchvision.models as models
if __name__ == '__main__':
    # model = YOLO(
    #     "E:/yolo/ultralytics-main_download/ultralytics/cfg/models/v8/yolov8n-RFCAC2F_EMA.yaml")
    # model.train(cfg="E:/yolo/ultralytics-main_download/ultralytics/cfg/myconfig.yaml",
    #             data="/E:/yolo/ultralytics-main_download/data.yaml", project="Results_Rotation", name="yolov8-RFCAC2F_EMAWIOU", resume=True)

# success = YOLO("/root/autodl-tmp/ultralytics-main/yolov8n.pt").export(format="onnx")


#预测

    # model = YOLO("E:/yolo/ultralytics-main_download/best.pt")  
    model = YOLO("E:/yolo/ultralytics-main_download/YOLOv8n.pt")  
    image = "E:/yolo/ultralytics-main_download/ultralytics/assets/"  # 图片文件夹
    model.predict(image, save=True)  # 返回image的预测结果

# 验证
# if __name__ == '__main__':

#     model = YOLO("E:/yolo/ultralytics-main_download/RE-YOLOv8.pt")
#     results=model.val(data="E:/yolo/ultralytics-main_download/data.yaml",imgsz=640,split='test',batch=1,conf=0.001,iou=0.5,name="re-yolo",optimizer='Adam')
    # with torch.cuda.device(0):
        # net = #models.get_model("E:/yolo/ultralytics-main_download/best.pt")
    # flops,params=get_model_complexity_info(model,input_res=(3,640,640),as_strings=True, print_per_layer_stat=False)
    # print(flops)