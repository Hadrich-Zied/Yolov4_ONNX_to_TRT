import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from tensorrt_inference.trt_inf_utils import letterbox,read_images
import numpy as np

# For reading size information ftch.shaperom batches
import struct
def get_batchs(dir_path,batch_sz=2,img_sz=640):
    path_list=[dir_path+'/'+i for i in os.listdir(dir_path)]
    n=len(path_list)//batch_sz
    path_list=path_list[:n*batch_sz]
    batches= []
    while path_list != []:
        img, shapes, paths, path_list = read_images(batch_sz, path_list,img_sz)
        img = np.transpose(img, (0, 3, 1, 2))
        img /= 255.0
        img = np.ascontiguousarray(img, dtype=np.float32)
        batches.append(img)
    batches=iter(batches)
    return batches
class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dir_path, cache_file,batch_sz=2,img_sz=640):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_sz=batch_sz
        self.cache_file = cache_file
        self.img_sz=img_sz
        # Get a list of all the batch files in the batch folder.
        self.batches=get_batchs(dir_path,batch_sz,self.img_sz)
        self.calibration_data = np.zeros((batch_sz, 3, self.img_sz, self.img_sz),dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.calibration_data.nbytes)

    def get_batch_size(self):
        return self.batch_sz

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Get a single batch.
            data = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print("_________________")
        print("calibration finished")
        print("_________________")
