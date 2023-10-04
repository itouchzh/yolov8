from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
 
# accumulate predictions from all images
# 载入coco2017验证集标注文件
coco_true = COCO(annotation_file="/root/code/ultralytics/data/instances_val2017_1.json")
# 载入网络在coco2017验证集上预测的结果
coco_pre = coco_true.loadRes('/root/code/ultralytics/runs/detect/val4/predictions.json')
 
coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()



# import json

# # 读取原始JSON文件
# with open('instances_val2017_t.json', 'r') as file:
#     data = json.load(file)

# # 遍历annotations列表，将category_id属性的值加1
# for annotation in data['annotations']:
#     annotation['category_id'] += 1

# # 保存修改后的数据到新的JSON文件
# with open('output.json', 'w') as file:
#     json.dump(data, file)
