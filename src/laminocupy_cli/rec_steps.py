from laminocupy_cli import backprojection
from laminocupy_cli import fbp_filter
from laminocupy_cli import retrieve_phase
from laminocupy_cli import remove_stripe
from laminocupy_cli import utils
from laminocupy_cli import logging
from cupyx.scipy.fft import rfft, irfft
import cupy as cp
import numpy as np
import numexpr as ne
import dxchange
import threading
import h5py
import os
import signal

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

log = logging.getLogger(__name__)


class GPURecSteps():
    """Main class for laminographic reconstruction with preprocessing procedures"""

    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        # retrieve data sizes
        with h5py.File(args.file_name) as fid:
            # determine sizes
            ni = fid['/exchange/data'].shape[2]
            ndark = fid['/exchange/data_dark'].shape[0]
            nflat = fid['/exchange/data_white'].shape[0]
            theta = fid['/exchange/theta'][:].astype('float32')/180*np.pi
            deth = fid['/exchange/data'].shape[1]

        # init lamino angle
        phi = np.pi/2-args.lamino_angle/180*np.pi
        # take reconstruction height
        nz = args.recon_height
        if nz == 0:
            nz = int(np.ceil((deth/np.sin(phi))/2**args.binning/4)) * \
                2**args.binning*4
        # define chunk size for processing
        ncz = args.nsino_per_chunk
        ncproj = args.nproj_per_chunk
        # take center
        centeri = args.rotation_axis
        if centeri == -1:
            centeri = ni/2
        # update sizes wrt binning
        ni //= 2**args.binning
        centeri /= 2**args.binning
        deth //= 2**args.binning
        nz //= 2**args.binning

        # change sizes for 360 deg scans with rotation axis at the border (not implemented yet): TODO
        n = ni
        center = centeri

        # blocked views fix
        ids_proj = np.arange(len(theta))
        theta = theta[ids_proj]

        if args.blocked_views:
            st = args.blocked_views_start
            end = args.blocked_views_end
            ids = np.where(((theta) % np.pi < st) +
                           ((theta-st) % np.pi > end-st))[0]
            theta = theta[ids]
            ids_proj = ids_proj[ids]
        nproj = len(theta)

        # c++ wrapper for fbp filter
        self.cl_filter = fbp_filter.FBPFilter(ni, ncproj, deth)

        self.n = n
        self.deth = deth
        self.nz = nz
        self.ncz = ncz
        self.nproj = nproj
        self.ncproj = ncproj
        self.center = center
        self.ni = ni
        self.centeri = centeri
        self.ndark = ndark
        self.nflat = nflat
        self.ids_proj = ids_proj
        self.theta = theta
        self.phi = phi
        self.args = args

    def downsample(self, data):
        """Downsample data"""

        data = data.astype('float32')
        for j in range(self.args.binning):
            x = data[:, :, ::2]
            y = data[:, :, 1::2]
            data = ne.evaluate('x + y')  # should use multithreading
        for k in range(self.args.binning):
            x = data[:, ::2]
            y = data[:, 1::2]
            data = ne.evaluate('x + y')
        return data

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""

        dark0 = cp.mean(dark.astype('float32'), axis=0)
        flat0 = cp.mean(flat.astype('float32'), axis=0)
        data = (data.astype('float32')-dark0)/(flat0-dark0)
        return data

    def minus_log(self, data):
        """Taking negative logarithm"""

        data = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0
        return data

    def fbp_filter_center(self, data):
        """FBP filtering of projections"""

        # size for padding
        ne = 3*self.n//2
        # init filter
        t = cp.fft.rfftfreq(ne).astype('float32')
        if self.args.fbp_filter == 'parzen':
            w = t * (1 - t * 2)**3
        elif self.args.fbp_filter == 'shepp':
            w = t * cp.sinc(t)
        # for moving the rotation center in the frequency domain
        # w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sh+self.n/2))  # center fix-> moved to backproj
        w = w.astype('complex64')
        # padding
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')

        # call c++ wrapper, having arrays as c contiguous
        data = cp.ascontiguousarray(data)
        w = cp.ascontiguousarray(w.view('float32'))
        # works faster and use less memory than the following
        self.cl_filter.filter(data, w, cp.cuda.get_current_stream())
        # data = irfft(w*rfft(data, axis=2), axis=2)

        # unpadding
        data = data[:, :, ne//2-self.n//2:ne//2+self.n//2]
        return data

    def proc_sino(self, res, data, dark, flat):
        """Processing a sinogram data chunk"""

        # dark flat field correrction
        data = self.darkflat_correction(data, dark, flat)
        # remove stripes
        if(self.args.remove_stripe_method == 'fw'):
            data = remove_stripe.remove_stripe_fw(
                data, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
        res[:] = data

    def proc_proj(self, res, data):
        """Processing a projection data chunk"""

        # retrieve phase
        if(self.args.retrieve_phase_method == 'paganin'):
            data = retrieve_phase.paganin_filter(
                data,  self.args.pixel_size*1e-4, self.args.propagation_distance/10, self.args.energy, self.args.retrieve_phase_alpha)
        # minus log
        data = self.minus_log(data)
        # fbp filter and compensation for the center shift
        data = self.fbp_filter_center(data)
        res[:] = data

    def read_data(self, data, fdata, k, lchunk):
        """Read a chunk of projection with binning"""

        d = fdata[k*lchunk:self.args.start_proj +
                  (k+1)*lchunk, :].astype('float32')
        data[k*lchunk:self.args.start_proj+(k+1)*lchunk] = self.downsample(d)

    def read_data_parallel(self, nthreads=8):
        """Read data in parallel chunks (good for SSD disks)"""

        fid = h5py.File(self.args.file_name, 'r')

        # read dark and flat, binning
        dark = fid['exchange/data_dark'][:].astype('float32')
        flat = fid['exchange/data_white'][:].astype('float32')
        flat = self.downsample(flat)
        dark = self.downsample(dark)

        # parallel read of projections
        data = np.zeros([self.nproj, self.deth, self.ni],
                        dtype='float32')
        lchunk = int(np.ceil(self.nproj/nthreads))
        threads = []
        for k in range(nthreads):
            read_thread = threading.Thread(target=self.read_data, args=(
                data, fid['exchange/data'], k, lchunk))
            threads.append(read_thread)
            read_thread.start()
        for thread in threads:
            thread.join()

        return data, dark, flat

    def write_h5(self, data, rec_dataset, start):
        """Save reconstruction chunk to an hdf5"""
        rec_dataset[start:start+data.shape[0]] = data


############################################### MAIN FUNCTIONS ###############################################

    def recon_steps(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps """

        log.info('Step 1. Reading data.')
        data, dark, flat = self.read_data_parallel()
        log.info('Step 2. Processing by chunks in z.')
        data = self.proc_sino_parallel(data, dark, flat)
        log.info('Step 3. Processing by chunks in theta.')
        data = self.proc_proj_parallel(data)

        log.info('Step 4. Reconstruction by chunks in z and theta.')
        self.rec_parallel(data)

    def recon_steps_try(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps """

        log.info('Step 1. Reading data.')
        data, dark, flat = self.read_data_parallel()

        log.info('Step 2. Processing by chunks in z.')
        data = self.proc_sino_parallel(data, dark, flat)

        log.info('Step 3. Processing by chunks in theta.')
        data = self.proc_proj_parallel(data)

        shift_array = np.arange(-self.args.center_search_width,
                                self.args.center_search_width, self.args.center_search_step*2**self.args.binning).astype('float32')/2**self.args.binning
        rslice = int(self.args.nsino*self.nz)
        log.info(f'Step 4. Reconstruction of the slice {rslice} for centers {self.center+shift_array}')
        log.info(f'{self.center+shift_array}')

        fnameout = os.path.dirname(
            self.args.file_name)+'_rec/try_center/'+os.path.basename(self.args.file_name)[:-3]+'/recon_'
        log.info(f'Output: {fnameout}')
        write_threads = []
        rec = self.rec_parallel_try(data, rslice, shift_array)
        dxchange.write_tiff(
            rec[0], f'{fnameout}{((self.centeri-shift_array[0])*2**self.args.binning):08.2f}', overwrite=True)
        for k in range(1, len(shift_array)):
            # avoid simultaneous directory creation
            write_thread = threading.Thread(target=dxchange.write_tiff,
                                            args=(rec[k],),
                                            kwargs={'fname': f'{fnameout}{((self.centeri+shift_array[k])*2**self.args.binning):08.2f}',
                                                    'overwrite': True})
            write_threads.append(write_thread)
            write_thread.start()

        for thread in write_threads:
            thread.join()
############################################### Parallel conveyor execution #############################################

    def proc_sino_parallel(self, data, dark, flat):

        res = np.zeros(data.shape, dtype='float32')

        nchunk = int(np.ceil(self.deth/self.ncz))
        lchunk = np.minimum(
            self.ncz, np.int32(self.deth-np.arange(nchunk)*self.ncz))  # chunk sizes

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.ni], dtype='float32'))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, self.ndark, self.ncz, self.ni], dtype='float32'))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, self.nflat, self.ncz, self.ni], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.nproj, self.ncz, self.ni], dtype='float32')
        item_gpu['dark'] = cp.zeros(
            [2, self.ndark, self.ncz, self.ni], dtype='float32')
        item_gpu['flat'] = cp.ones(
            [2, self.nflat, self.ncz, self.ni], dtype='float32')

        # pinned memory for res
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.n], dtype='float32'))
        # gpu memory for res
        rec = cp.zeros([2, self.nproj, self.ncz, self.n], dtype='float32')

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, nchunk-k+1, length=40)

            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.proc_sino(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2],
                                   item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2])
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :, :lchunk[k]
                                    ] = data[:, k*self.ncz:k*self.ncz+lchunk[k]]
                item_pinned['dark'][k % 2, :, :lchunk[k]
                                    ] = dark[:, k*self.ncz:k*self.ncz+lchunk[k]]
                item_pinned['flat'][k % 2, :, :lchunk[k]
                                    ] = flat[:, k*self.ncz:k*self.ncz+lchunk[k]]
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            stream3.synchronize()
            if(k > 1):
                res[:, (k-2)*self.ncz:(k-2)*self.ncz+lchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :, :lchunk[k-2]].copy()
            stream1.synchronize()
            stream2.synchronize()
        return res

    def proc_proj_parallel(self, data):

        res = np.zeros(data.shape, dtype='float32')

        nchunk = int(np.ceil(self.nproj/self.ncproj))
        lchunk = np.minimum(
            self.ncproj, np.int32(self.nproj-np.arange(nchunk)*self.ncproj))  # chunk sizes

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.deth, self.n], dtype='float32'))
        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.deth, self.n], dtype='float32')

        # pinned memory for processed data
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncproj, self.deth, self.n], dtype='float32'))
        # gpu memory for processed data
        rec = cp.zeros([2, self.ncproj, self.deth, self.n], dtype='float32')

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, nchunk-k+1, length=40)
            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.proc_proj(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2])

            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :lchunk[k]
                                    ] = data[self.ncproj*k:self.ncproj*k+lchunk[k]]
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])

            stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                res[(k-2)*self.ncproj:(k-2)*self.ncproj+lchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :lchunk[k-2]].copy()
            stream1.synchronize()
            stream2.synchronize()
        return res

    def rec_parallel(self, data):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        nzchunk = int(np.ceil(self.nz/self.ncz))
        lzchunk = np.minimum(
            self.ncz, np.int32(self.nz-np.arange(nzchunk)*self.ncz))  # chunk sizes in z
        ntchunk = int(np.ceil(self.nproj/self.ncproj))
        ltchunk = np.minimum(
            self.ncproj, np.int32(self.nproj-np.arange(ntchunk)*self.ncproj))  # chunk sizes in proj
        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.deth, self.ni], dtype='float32'))
        item_pinned['theta'] = utils.pinned_array(
            np.zeros([2, self.ncproj], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.deth, self.ni], dtype='float32')
        item_gpu['theta'] = cp.zeros(
            [2, self.ncproj], dtype='float32')

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype='float32'))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype='float32')

        # list of threads for parallel writing to hard disk
        write_threads = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        if self.args.save_format == 'tiff':
            if(self.args.out_path_name is None):
                fnameout = os.path.dirname(
                    self.args.file_name)+'_rec/'+os.path.basename(self.args.file_name)[:-3]+'_rec/recon'
            else:
                fnameout = str(self.args.out_path_name)+'/r'
        elif self.args.save_format == 'h5':
            if not os.path.isdir(os.path.dirname(self.args.file_name)+'_rec'):
                os.mkdir(os.path.dirname(self.args.file_name)+'_rec')
            if(self.args.out_path_name is None):
                fnameout = os.path.dirname(
                    self.args.file_name)+'_rec/'+os.path.basename(self.args.file_name)[:-3]+'_rec.h5'
            else:
                fnameout = str(self.args.out_path_name)
            os.system(f'rm -rf {fnameout}')
            fid_rec = h5py.File(fnameout, 'w')
            sid = '/exchange/recon'
            rec_dataset = fid_rec.create_dataset(sid, shape=(
                self.nz, self.n, self.n), chunks=(1, self.n, self.n), dtype='float32')

        # Conveyor for data cpu-gpu copy and reconstruction
        for kz in range(nzchunk+2):
            utils.printProgressBar(kz, nzchunk+1, nzchunk-kz+1, length=40)
            rec[(kz-1) % 2][:] = 0
            for kt in range(ntchunk+2):
                if (kz > 0 and kz < nzchunk+1 and kt > 0 and kt < ntchunk+1):
                    with stream2:  # reconstruction
                        backprojection.adj(rec[(kz-1) % 2],
                                           item_gpu['data'][(kt-1) % 2],
                                           item_gpu['theta'][(kt-1) % 2],
                                           self.center, self.phi, (kz-1)*self.ncz-self.nz//2)
                if (kz > 1 and kt == 0):
                    with stream3:  # gpu->cpu copy
                        rec[(kz-2) % 2].get(out=rec_pinned[(kz-2) % 2])
                if(kt < ntchunk):
                    # copy to pinned memory
                    item_pinned['data'][kt % 2][:ltchunk[kt]
                                                ] = data[kt*self.ncproj:kt*self.ncproj+ltchunk[kt]]
                    item_pinned['theta'][kt % 2][:ltchunk[kt]
                                                 ] = self.theta[kt*self.ncproj:kt*self.ncproj+ltchunk[kt]]
                    item_pinned['data'][kt % 2][ltchunk[kt]:] = 0
                    with stream1:  # cpu->gpu copy
                        item_gpu['data'][kt % 2].set(
                            item_pinned['data'][kt % 2])
                        item_gpu['theta'][kt % 2].set(
                            item_pinned['theta'][kt % 2])
                stream3.synchronize()
                if (kz > 1 and kt == 0):
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    rec_pinned0 = rec_pinned[(kz-2) % 2, :lzchunk[kz-2]].copy()
                    if self.args.save_format == 'tiff':
                        write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                        args=(rec_pinned0,),
                                                        kwargs={'fname': fnameout,
                                                                'start':  (kz-2)*self.ncz,
                                                                'overwrite': True})
                    elif self.args.save_format == 'h5':
                        write_thread = threading.Thread(target=self.write_h5,
                                                        args=(rec_pinned0, rec_dataset, (kz-2)*self.ncz))

                    write_threads.append(write_thread)
                    write_thread.start()
                stream1.synchronize()
                stream2.synchronize()
        log.info(f'Output: {fnameout}')
        # wait until reconstructions are written to hard disk
        for thread in write_threads:
            thread.join()

    def rec_parallel_try(self, data, rslice, sh):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        nzchunk = int(np.ceil(len(sh)/self.ncz))
        lzchunk = np.minimum(
            self.ncz, np.int32(len(sh)-np.arange(nzchunk)*self.ncz))  # chunk sizes in z
        ntchunk = int(np.ceil(self.nproj/self.ncproj))
        ltchunk = np.minimum(
            self.ncproj, np.int32(self.nproj-np.arange(ntchunk)*self.ncproj))  # chunk sizes in proj
        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(                    
            np.zeros([2, self.ncproj, self.deth, self.ni], dtype='float32'))
        item_pinned['theta'] = utils.pinned_array(
            np.zeros([2, self.ncproj], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.deth, self.ni], dtype='float32')
        item_gpu['theta'] = cp.zeros(
            [2, self.ncproj], dtype='float32')

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype='float32'))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype='float32')

        cen = cp.zeros([nzchunk*self.ncz], dtype='float32')
        cen[:len(sh)] = cp.array(sh)+self.center

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        res = np.zeros([len(sh), self.n, self.n], dtype='float32')
        # Conveyor for data cpu-gpu copy and reconstruction
        for kz in range(nzchunk+2):
            rec[(kz-1) % 2][:] = 0

            for kt in range(ntchunk+2):
                if (kz > 0 and kz < nzchunk+1 and kt > 0 and kt < ntchunk+1):
                    with stream2:  # reconstruction
                        backprojection.adj_try(rec[(kz-1) % 2],
                                               item_gpu['data'][(kt-1) % 2],
                                               item_gpu['theta'][(kt-1) % 2],
                                               cen[(kz-1)*self.ncz:kz*self.ncz],
                                               self.phi, rslice-self.nz//2)
                if (kz > 1 and kt == 0):
                    with stream3:  # gpu->cpu copy
                        rec[(kz-2) % 2].get(out=rec_pinned[(kz-2) % 2])
                if(kt < ntchunk):
                    # copy to pinned memory
                    item_pinned['data'][kt % 2][:ltchunk[kt]
                                                ] = data[kt*self.ncproj:kt*self.ncproj+ltchunk[kt]]
                    item_pinned['theta'][kt % 2][:ltchunk[kt]
                                                 ] = self.theta[kt*self.ncproj:kt*self.ncproj+ltchunk[kt]]
                    item_pinned['data'][kt % 2][ltchunk[kt]:] = 0
                    with stream1:  # cpu->gpu copy
                        item_gpu['data'][kt % 2].set(
                            item_pinned['data'][kt % 2])
                        item_gpu['theta'][kt % 2].set(
                            item_pinned['theta'][kt % 2])
                stream3.synchronize()
                if (kz > 1 and kt == 0):
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    res[(kz-2)*self.ncz:(kz-2)*self.ncz+lzchunk[kz-2]
                        ] = rec_pinned[(kz-2) % 2, :lzchunk[kz-2]].copy()
                stream1.synchronize()
                stream2.synchronize()
        return res
