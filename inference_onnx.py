import os
from tensorrt_inference.trt_inf_utils import nms,scale_coords,read_images
import numpy as np
import onnxruntime
import time
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', type=str, default='/home/ai/PycharmProjects/Inference/onnx_engines/yolov4-p5-640.onnx', help='onnx path')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--img-size', type=int, default=640, help='img size')
parser.add_argument('--pathtxt', type=str, default="/home/ai/Desktop/DATASET_INSULATORS/yolo_test/test_yolo.txt", help='txt path')
parser.add_argument('--savetxt', action='store_true', default=False, help='save txt')
parser.add_argument('--savexml', action='store_true', default=False, help='save xml')
parser.add_argument('--output_dir', type=str, default="/home/ai/PycharmProjects/Inference/onnx_inf/", help='output dir')
parser.add_argument('--half', action='store_true', default=False, help='FP16')
opt = parser.parse_args()

onnx_pth,batch_size,pathtxt,savetxt,outputdir,half=opt.onnx,opt.batch_size,opt.pathtxt,opt.savetxt,opt.output_dir,opt.half
savexml=opt.savexml
img_sz=opt.img_size
providers = ['CUDAExecutionProvider']
session=onnxruntime.InferenceSession(onnx_pth,providers=providers)
session.get_modelmeta()
input=session.get_inputs()[0].name
output=session.get_outputs()[0].name
sh=session.get_inputs()[0].shape
img=np.zeros(sh,dtype=np.float32) if not half else np.zeros(sh,dtype=np.half)
result = session.run(["output"],{"images":img})

classes=["insulator"]
f=open(pathtxt,'r')
path_list=[l.rstrip('\n') for l in f.readlines()]
inf_time=[]
j=0
while path_list!=[]:

    img, shapes, paths, path_list=read_images(batch_size,path_list,img_sz)
    img_r=img[:]
    img=np.transpose(img,(0,3,1,2))
    img /= 255.0
    img = np.ascontiguousarray(img,dtype=np.float32) if not half else  np.ascontiguousarray(img,dtype=np.half)
    t = time.time()
    result = session.run(["output"],{"images":img})
    t0=time.time() - t

    result = result[-1].reshape(2, -1, 6)
    t = time.time()
    output=nms(result,0.5,0.6)
    t1 = time.time()- t

    inf_time.append((t0+t1)/2)
    #print("inference after nms %.4f" % (t0+t1)/2)

    if(j%100==0):
        print("running ...")
        print('model inference time : ', t0 * 1000 / 2)
        print('nms time : ', t1 * 1000 / 2)
    j= j + 1

    if savetxt:
        out_d=onnx_pth.split('/')[-1].split('.')[0]
        out_d=outputdir+'/'+out_d+'/'
        os.makedirs(out_d, exist_ok=True)
        for i, det in enumerate(output):
            img_name=paths[i].split('/')[-1]
            img_name=img_name.split('.')[0]
            txt_pth=out_d+img_name+'.txt'
            f = open(txt_pth, 'w')
            if(det!=[]):
                det[:,:4]=scale_coords(img_r[i].shape,det[:,:4],shapes[i])
                for d in det:
                    x1,y1,x2,y2,conf,cls =d[0],d[1],d[2],d[3],d[4],d[5]
                    f.write(('%g '*6+' \n') % (cls,conf,x1,y1,x2,y2))
        f.close()
    if savexml:
        out_d=onnx_pth.split('/')[-1].split('.')[0]
        out_d=outputdir+'/'+out_d+'/'
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
print('average inference time en ms')
print(sum(inf_time)/len(inf_time)*1000)
