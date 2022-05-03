from laminocupy_cli import cfunc_filter
import cupy as cp


class FBPFilter():
    def __init__(self, n, ntheta, deth):
        self.fslv = cfunc_filter.cfunc_filter(ntheta, deth, n)

    def filter(self, data, w, stream):
        # reorganize data as a complex array, reuse data
        data = cp.ascontiguousarray(data)
        w = cp.ascontiguousarray(w.view('float32'))
        self.fslv.filter(data.data.ptr, w.data.ptr, stream.ptr)
