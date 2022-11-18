import tensorrt as trt
from calibrator import PythonEntropyCalibrator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', type=str, default='onnx_engines/yolov4-tiny-640.onnx', help='onnx path')
parser.add_argument('--img-size', type=int, default=640, help='img size')
parser.add_argument('--trt-eng', type=str, default="trt_engines/yolov4-tiny-640.trt", help='trt engine save path')

parser.add_argument('--precision', type=str, default='FP32', help='FP32,FP16,INT8')
parser.add_argument('--cal_dir', type=str, default="/home/ai/Desktop/DATASET_INSULATORS/train_merged/JPEGImages/", help='calibration dir path')
parser.add_argument('--cache', type=str, default="caches/yolov4-tiny-640.bin", help='cahe file path')
opt = parser.parse_args()

onnx,img_sz,trt_eng=opt.onnx,opt.img_size,opt.trt_eng
precision=opt.precision
cal_dir,cache=opt.cal_dir,opt.cache



logger=trt.Logger(trt.Logger.WARNING)
builder=trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser=trt.OnnxParser(network,logger)
sucess=parser.parse_from_file(onnx)
for idx in range (parser.num_errors):
    print(parser.get_error(idx))
config=builder.create_builder_config()
config.max_workspace_size=(1<<30)
config.set_flag(trt.BuilderFlag.DEBUG)
if (precision=='FP32'):
    pass
elif(precision=='FP16'):
    config.set_flag(trt.BuilderFlag.FP16)
elif(precision=='INT8'):
    int8_calibrator = PythonEntropyCalibrator(cal_dir, cache, 8,img_sz=img_sz)
    config.int8_calibrator = int8_calibrator
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
#config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE , 1<<20)
serialized_eng=builder.build_serialized_network(network,config)
with open(trt_eng, 'wb') as f:
    f.write(serialized_eng)
