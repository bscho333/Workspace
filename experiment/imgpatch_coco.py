# # Dataset Preparation
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="liuhaotian/llava-v1.5-7b", local_dir='/home/bscho333/Workspace/data/llava')


from pycocotools.coco import COCO
import json

dataDir='/home/bscho333/data/coco'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# coco=COCO(annFile)
# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print(len(nms))

coco=COCO(annFile)
