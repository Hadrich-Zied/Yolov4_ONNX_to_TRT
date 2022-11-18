#!/usr/bin/python3

# ===================================================================
#    IMPORT SECTION
# ===================================================================
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np


# ===================================================================
#    FUNCTION DEFINITION
# ===================================================================


# ===================================================================
#    CLASS DEFINITION
# ===================================================================

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:

    """
    This class is copied from:
    https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python
    and follows instruction provided in the "NVIDIA TensorRT Developer Guide"
    """

    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        self.ctx=cuda.Device(0).make_context()
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.engine_batch_size = self.engine.max_batch_size
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) # * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size=1):
        self.ctx.push()
        hinp=self.inputs
        hout=self.outputs
        bind=self.bindings
        st=self.stream

        x = x.astype(self.dtype)

        np.copyto(hinp[0].host, x.ravel())
        # np.copyto(self.inputs[1].host, x.ravel())

        for inp in hinp:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        self.context.execute_async(batch_size=self.engine_batch_size, bindings=bind, stream_handle=st.handle)
        for out in hout:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        st.synchronize()
        self.ctx.pop()
        # return [out.host.reshape(batch_size, -1) for out in self.outputs]
        return [out.host.reshape(batch_size, -1) for out in hout]


# ===================================================================
#    START OF PROGRAM
# ===================================================================
if __name__ == '__main__':
    pass
