//! Typed kernel handles for compile-time-verified GPU kernel launches.
//!
//! This module provides [`Kernel`], a typed wrapper around [`Function`] that encodes the
//! kernel's parameter signature in the type system. This eliminates an entire class of
//! runtime errors (wrong argument count or type) and makes kernel launch sites self-documenting.
//!
//! # Quick example
//!
//! ```no_run
//! use cust::prelude::*;
//! use cust::kernel::Kernel;
//!
//! # fn demo(module: &Module, stream: &Stream) -> cust::error::CudaResult<()> {
//! // Load a kernel with a known signature: two f32 device pointers, a scalar, and a length.
//! let saxpy: Kernel<(DevicePointer<f32>, DevicePointer<f32>, f32, usize)> =
//!     module.get_kernel("saxpy")?;
//!
//! // Launch is type-checked: the tuple must exactly match the Kernel's Args type.
//! unsafe {
//!     saxpy.launch(256, 128, 0, stream, (a_ptr, b_ptr, 2.0f32, 1024usize))?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Design
//!
//! - `Kernel<'a, Args>` carries the argument tuple as a phantom type parameter.
//! - `KernelArgs` is an `unsafe` sealed trait implemented for tuples of [`DeviceCopy`] types.
//! - The maximum arity is 12, which covers virtually all real-world kernels.
//! - Occupancy queries and attributes are forwarded to the underlying [`Function`].

use std::ffi::c_void;
use std::marker::PhantomData;

use crate::error::CudaResult;
use crate::function::{BlockSize, Function, FunctionAttribute, GridSize};
use crate::memory::DeviceCopy;
use crate::module::Module;
use crate::stream::Stream;

mod private {
    pub trait Sealed {}
}

/// Trait for types that can be passed as kernel arguments.
///
/// This is implemented for tuples of [`DeviceCopy`] types up to 12 elements.
/// The trait is `unsafe` because incorrect implementations could cause the kernel
/// to read garbage parameter values.
///
/// # Safety
/// Implementors must ensure that [`KernelArgs::write_ptrs`] produces pointers that
/// are valid for the duration of the kernel launch (i.e. until `cuLaunchKernel` returns).
pub unsafe trait KernelArgs: private::Sealed {
    /// Write the addresses of each argument element into `out`.
    ///
    /// `out` is guaranteed to have at least [`KernelArgs::LEN`] slots.
    fn write_ptrs(&self, out: &mut [*mut c_void]);

    /// Number of arguments in this tuple.
    const LEN: usize;
}

impl private::Sealed for () {}
unsafe impl KernelArgs for () {
    fn write_ptrs(&self, _out: &mut [*mut c_void]) {}
    const LEN: usize = 0;
}

macro_rules! impl_kernel_args {
    // helper: count tokens
    (@count) => { 0usize };
    (@count $x:tt $($rest:tt)*) => { 1usize + impl_kernel_args!(@count $($rest)*) };

    ($($ty:ident),+) => {
        impl<$($ty: DeviceCopy),+> private::Sealed for ($($ty,)+) {}
        unsafe impl<$($ty: DeviceCopy),+> KernelArgs for ($($ty,)+) {
            fn write_ptrs(&self, out: &mut [*mut c_void]) {
                #[allow(non_snake_case)]
                let ($($ty,)+) = self;
                let mut _i = 0usize;
                $(
                    out[_i] = $ty as *const _ as *mut c_void;
                    _i += 1;
                )+
            }

            const LEN: usize = impl_kernel_args!(@count $($ty)+);
        }
    };
}

impl_kernel_args!(A0);
impl_kernel_args!(A0, A1);
impl_kernel_args!(A0, A1, A2);
impl_kernel_args!(A0, A1, A2, A3);
impl_kernel_args!(A0, A1, A2, A3, A4);
impl_kernel_args!(A0, A1, A2, A3, A4, A5);
impl_kernel_args!(A0, A1, A2, A3, A4, A5, A6);
impl_kernel_args!(A0, A1, A2, A3, A4, A5, A6, A7);
impl_kernel_args!(A0, A1, A2, A3, A4, A5, A6, A7, A8);
impl_kernel_args!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9);
impl_kernel_args!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
impl_kernel_args!(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11);

/// A compile-time-typed handle to a kernel function loaded from a [`Module`].
///
/// `Args` is a tuple type describing the kernel's parameter signature. Each element of the tuple
/// must implement [`DeviceCopy`].
///
/// # Example
///
/// ```no_run
/// use cust::prelude::*;
/// use cust::kernel::Kernel;
///
/// # fn demo(module: &Module, stream: &Stream) -> cust::error::CudaResult<()> {
/// let vecadd: Kernel<(DevicePointer<f32>, usize, DevicePointer<f32>, usize, DevicePointer<f32>)> =
///     module.get_kernel("vecadd")?;
///
/// unsafe {
///     vecadd.launch(256, 128, 0, stream, (a, a_len, b, b_len, c))?;
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Kernel<'a, Args: KernelArgs> {
    func: Function<'a>,
    _marker: PhantomData<Args>,
}

unsafe impl<Args: KernelArgs> Send for Kernel<'_, Args> {}
unsafe impl<Args: KernelArgs> Sync for Kernel<'_, Args> {}

impl<'a, Args: KernelArgs> Kernel<'a, Args> {
    /// Load a kernel by name from a module, typed by the expected argument signature.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Module::get_function`] (e.g. `NotFound` if the
    /// symbol does not exist in the module).
    pub fn from_module(module: &'a Module, name: &str) -> CudaResult<Self> {
        let func = module.get_function(name)?;
        Ok(Self {
            func,
            _marker: PhantomData,
        })
    }

    /// Launch the kernel with the given grid/block configuration and arguments.
    ///
    /// # Safety
    ///
    /// Launching kernels is inherently unsafe: the kernel runs on the device and may
    /// access memory in ways not checked by the Rust type system. The caller must
    /// ensure that:
    /// - The kernel was compiled for the current device's compute capability.
    /// - Device pointers passed as arguments are valid and accessible.
    /// - The host does not access device/unified memory that the kernel writes to
    ///   until after `stream.synchronize()`.
    pub unsafe fn launch<G, B>(
        &self,
        grid: G,
        block: B,
        shared_mem: u32,
        stream: &Stream,
        args: Args,
    ) -> CudaResult<()>
    where
        G: Into<GridSize>,
        B: Into<BlockSize>,
    {
        let mut ptrs = [std::ptr::null_mut::<c_void>(); 12];
        args.write_ptrs(&mut ptrs[..Args::LEN]);
        unsafe { stream.launch(&self.func, grid, block, shared_mem, &ptrs[..Args::LEN]) }
    }

    /// Returns information about the underlying function.
    ///
    /// See [`Function::get_attribute`] for details.
    pub fn get_attribute(&self, attr: FunctionAttribute) -> CudaResult<i32> {
        self.func.get_attribute(attr)
    }

    /// The maximum number of active blocks per SM for a given block size and dynamic shared memory.
    ///
    /// See [`Function::max_active_blocks_per_multiprocessor`] for details.
    pub fn max_active_blocks_per_multiprocessor(
        &self,
        block_size: BlockSize,
        dynamic_smem_size: usize,
    ) -> CudaResult<u32> {
        self.func
            .max_active_blocks_per_multiprocessor(block_size, dynamic_smem_size)
    }

    /// Returns a reasonable block and grid size to achieve maximum occupancy.
    ///
    /// See [`Function::suggested_launch_configuration`] for details.
    pub fn suggested_launch_configuration(
        &self,
        dynamic_smem_size: usize,
        block_size_limit: BlockSize,
    ) -> CudaResult<(u32, u32)> {
        self.func
            .suggested_launch_configuration(dynamic_smem_size, block_size_limit)
    }

    /// The amount of dynamic shared memory available per block for a given launch configuration.
    ///
    /// See [`Function::available_dynamic_shared_memory_per_block`] for details.
    pub fn available_dynamic_shared_memory_per_block(
        &self,
        blocks: GridSize,
        block_size: BlockSize,
    ) -> CudaResult<usize> {
        self.func
            .available_dynamic_shared_memory_per_block(blocks, block_size)
    }

    /// Access the raw [`Function`] handle.
    pub fn as_function(&self) -> &Function<'a> {
        &self.func
    }
}

impl<'a, Args: KernelArgs> From<Kernel<'a, Args>> for Function<'a> {
    fn from(kernel: Kernel<'a, Args>) -> Self {
        kernel.func
    }
}

/// Convenience macro to load a typed kernel from a module.
///
/// # Syntax
///
/// ```ignore
/// let kernel = typed_kernel!(module, "kernel_name" => (ArgType1, ArgType2, ...));
/// ```
///
/// # Example
///
/// ```no_run
/// use cust::prelude::*;
///
/// # fn demo(module: &Module) -> cust::error::CudaResult<()> {
/// let vecadd = typed_kernel!(module, "vecadd" => (
///     DevicePointer<f32>, usize,
///     DevicePointer<f32>, usize,
///     DevicePointer<f32>
/// ))?;
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! typed_kernel {
    ($module:expr, $name:expr => ($($arg_ty:ty),* $(,)?)) => {{
        $crate::kernel::Kernel::<($($arg_ty,)* )>::from_module($module, $name)
    }};
}
