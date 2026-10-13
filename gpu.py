#!/usr/bin/env python3
"""
TensorRT inference for ProInspect, with no third-party CUDA package.

There is no pip on this appliance and no pycuda in apt, so device memory is
managed through ctypes against libcudart directly. TensorRT 10 takes raw
integer device pointers via set_tensor_address, which is all the binding that
is needed - nothing to install, nothing to break on the next JetPack bump.

Engines are built by trtexec (models/build.sh) and are tied to this TensorRT
version and this GPU: a build artifact of the appliance, never copied between
boards and never committed.

    from gpu import TRTEngine
    rec = TRTEngine("models/ppocr_rec.engine")
    out = rec.infer(x)          # numpy in (any shape inside the profile), numpy out
"""

import ctypes
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODELS = HERE / "models"

H2D = 1                    # cudaMemcpyKind
D2H = 2


class CudaError(RuntimeError):
    pass


class _Cuda:
    """Minimal ctypes binding to the CUDA runtime: only what inference needs."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        # argtypes are not optional: without them ctypes truncates 64-bit
        # pointers and sizes on aarch64 and the failure looks like corruption.
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]

    def check(self, rc, what):
        if rc != 0:
            msg = self.lib.cudaGetErrorString(rc)
            raise CudaError("%s: %s" % (what, msg.decode() if msg else rc))

    def device_count(self):
        n = ctypes.c_int()
        if self.lib.cudaGetDeviceCount(ctypes.byref(n)) != 0:
            return 0
        return n.value

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(max(nbytes, 1))), "cudaMalloc")
        return p

    def free(self, p):
        if p:
            self.lib.cudaFree(p)

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s

    def sync(self, s):
        self.check(self.lib.cudaStreamSynchronize(s), "cudaStreamSynchronize")

    def to_device(self, dst, arr, stream):
        src = arr.ctypes.data_as(ctypes.c_void_p)
        self.check(self.lib.cudaMemcpyAsync(dst, src, ctypes.c_size_t(arr.nbytes), H2D, stream),
                   "cudaMemcpy H2D")

    def from_device(self, arr, src, nbytes, stream):
        dst = arr.ctypes.data_as(ctypes.c_void_p)
        self.check(self.lib.cudaMemcpyAsync(dst, src, ctypes.c_size_t(nbytes), D2H, stream),
                   "cudaMemcpy D2H")


_cuda = None


def cuda():
    global _cuda
    if _cuda is None:
        _cuda = _Cuda()
    return _cuda


def gpu_available():
    """True if a CUDA device is usable. Never raises."""
    try:
        return cuda().device_count() > 0
    except Exception:
        return False


class TRTEngine:
    """
    A loaded engine with device buffers sized for the largest shape in its
    optimisation profile, so any input inside the profile runs without a
    reallocation. Dynamic axes are set per call.
    """

    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        self.path = pathlib.Path(engine_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError("could not deserialize %s - built for another TensorRT or GPU?"
                               % self.path.name)
        self.ctx = self.engine.create_execution_context()
        self.c = cuda()
        self.stream = self.c.stream()
        self.inputs, self.outputs, self.devmem = [], [], {}

        # Pin every input to its profile maximum so the outputs report their
        # maximum shapes too; that sizes all the buffers once.
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                mn, opt, mx = self.engine.get_tensor_profile_shape(name, 0)
                self.ctx.set_input_shape(name, tuple(mx))
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(int(d) for d in self.ctx.get_tensor_shape(name))
            host = np.empty(shape, dtype=dtype)
            dev = self.c.malloc(host.nbytes)
            self.devmem[name] = dev
            self.ctx.set_tensor_address(name, int(dev.value))
            rec = {"name": name, "max_shape": shape, "dtype": dtype, "host": host}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                rec["min_shape"] = tuple(int(d) for d in self.engine.get_tensor_profile_shape(name, 0)[0])
                self.inputs.append(rec)
            else:
                self.outputs.append(rec)

    def infer(self, *arrays):
        """
        Run one inference. Positional arrays map to the engine's inputs in
        order and may take any shape inside the profile. Returns one output
        array (a list when there are several), shaped as the engine reports.
        """
        for rec, arr in zip(self.inputs, arrays):
            a = np.ascontiguousarray(arr, dtype=rec["dtype"])
            for d, lo, hi in zip(a.shape, rec["min_shape"], rec["max_shape"]):
                if d < lo or d > hi:
                    raise ValueError("%s: shape %s outside profile %s..%s"
                                     % (rec["name"], a.shape, rec["min_shape"], rec["max_shape"]))
            if not self.ctx.set_input_shape(rec["name"], tuple(a.shape)):
                raise ValueError("%s: set_input_shape(%s) refused" % (rec["name"], a.shape))
            self.c.to_device(self.devmem[rec["name"]], a, self.stream)
        if not self.ctx.execute_async_v3(int(self.stream.value)):
            raise RuntimeError("execute_async_v3 failed")
        outs = []
        for rec in self.outputs:
            shape = tuple(int(d) for d in self.ctx.get_tensor_shape(rec["name"]))
            n = int(np.prod(shape))
            flat = rec["host"].reshape(-1)
            self.c.from_device(flat, self.devmem[rec["name"]], n * flat.itemsize, self.stream)
            outs.append((flat, n, shape))
        self.c.sync(self.stream)
        results = [flat[:n].reshape(shape).copy() for flat, n, shape in outs]
        return results[0] if len(results) == 1 else results

    def benchmark(self, shape=None, iters=50, warmup=5):
        """Median latency in ms at the profile's optimum (or a given) shape."""
        rec = self.inputs[0]
        if shape is None:
            shape = tuple(self.engine.get_tensor_profile_shape(rec["name"], 0)[1])
        dummy = np.zeros(shape, dtype=rec["dtype"])
        for _ in range(warmup):
            self.infer(dummy)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            self.infer(dummy)
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        return {"shape": shape, "median_ms": round(ts[len(ts) // 2], 2),
                "p95_ms": round(ts[int(len(ts) * 0.95)], 2)}

    def close(self):
        for p in self.devmem.values():
            self.c.free(p)
        self.devmem.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    print("CUDA devices:", cuda().device_count())
    for path in sys.argv[1:] or sorted(MODELS.glob("*.engine")):
        e = TRTEngine(path)
        print(pathlib.Path(path).name)
        for r in e.inputs:
            print("  in  %-24s min %s max %s %s" % (r["name"], r["min_shape"], r["max_shape"], np.dtype(r["dtype"])))
        for r in e.outputs:
            print("  out %-24s max %s %s" % (r["name"], r["max_shape"], np.dtype(r["dtype"])))
        print("  ", e.benchmark())
        e.close()
