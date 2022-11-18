import os
import xml.etree.ElementTree as ET
from trt_inf_utils import read_images,nms,scale_coords
from trt_utils import TrtModel
import numpy as np
import pycuda
import time
import argparse
pycuda.autoinit

parser = argparse.ArgumentParser()
parser.add_argument('--trt', type=str, default='trt_engines/yolov4-640-fp16.trt', help='weights path')
parser.add_argument('--img-size',  type=int, default=640, help='image size')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--path', type=str, default="/home/ai/Desktop/DATASET_INSULATORS/yolo_test/test_yolo.txt", help='path')
parser.add_argument('--savexml', action="store_true", default=False, help='save')
parser.add_argument('--dir', type=str, default="trt_inf/" , help='dir')
opt = parser.parse_args()

engine_path=opt.trt
out_dir=opt.dir
savexml=opt.savexml
test_txt=opt.path
img_sz=opt.img_size
classes=["insulator"]
f=open(test_txt,'r')
path_list=[l.rstrip('\n') for l in f.readlines()]
print(len(path_list))
#parser = argparse.ArgumentParser()
#opt=parser.parse_args()
#opt.single_cls=True
#dataloader = create_dataloader(IMG_PATH, 640, 2, 64, opt, rect=False)[0]
#path_list=[IMG_PATH+ i for i in os.listdir(IMG_PATH)]
batch_size=2
t0,t1=0,0
inf_time=[]
model=TrtModel(engine_path,2)
j=0
while path_list!=[]:

    img, shapes, paths, path_list=read_images(batch_size,path_list,img_sz)
    img_r=img[:]
    img=np.transpose(img,(0,3,1,2))
    img /= 255.0
    img = np.ascontiguousarray(img)

    t = time.time()
    result = model(img, batch_size)
    t0=time.time() - t
    result=result[-1].reshape(2,-1,6)
    t=time.time()
    #output = non_max_suppression(torch.Tensor(result), conf_thres=0.5, iou_thres=0.6)
    output=nms(result,0.5,0.6)
    t1 = time.time()- t
    inf_time.append((t0+t1)/2)

    if(j%100==0):
        print("running ...")
        print('model inference time : ', t0 * 1000 / 2)
        print('nms time : ', t1 * 1000 / 2)
    j= j + 1
    #print("inference after nms %.2f" % (t0+t1)*1000)
    if savexml:
        out_d=engine_path.split('/')[-1].split('.')[0]
        out_d=out_dir+'/'+out_d+'/'
        os.makedirs(out_d, exist_ok=True)
        for i, det in enumerate(output):
            img_name=paths[i].split('/')[-1]
            img_name=img_name.split('.')[0]
            xml_pth=out_d+img_name+'.xml'
            annot = ET.Element('annotation')
            ET.SubElement(annot, "filename").text = img_name
            size = ET.SubElement(annot, "size")
            ET.SubElement(size, "width").text =str(shapes[i][0])
            ET.SubElement(size, "height").text = str(shapes[i][1])
            ET.SubElement(size, "depth").text = str(3)
            ET.SubElement(annot, "segmented").text = "0"
            owner = ET.SubElement(annot, "owner")
            ET.SubElement(owner, "name").text = "DRB SRL"
            if(det!=[]):
                det[:,:4]=scale_coords(img_r[i].shape,det[:,:4],shapes[i])
                for d in det:
                    x1,y1,x2,y2,conf,cls =d[0],d[1],d[2],d[3],d[4],d[5]
                    obj = ET.SubElement(annot, "object")
                    ET.SubElement(obj, "name").text = str(classes[int(cls)])
                    ET.SubElement(obj, "pose").text = "unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(x1)
                    ET.SubElement(bndbox, "xmax").text = str(x2)
                    ET.SubElement(bndbox, "ymin").text = str(y1)
                    ET.SubElement(bndbox, "ymax").text = str(y2)
                    ET.SubElement(obj, "confidence").text = str("{:.2f}".format(float(conf) * 100)) + "%"
            annotation = ET.ElementTree(annot)
            annotation.write(xml_pth)



print(sum(inf_time)/len(inf_time) * 1000)
model.ctx.pop()
del model.ctx

