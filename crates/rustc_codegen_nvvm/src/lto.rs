use std::ffi::CString;
use std::sync::Arc;

use rustc_codegen_ssa::{
    CompiledModule, ModuleCodegen,
    back::{
        lto::{SerializedModule, ThinModule, ThinShared},
        write::{CodegenContext, SharedEmitter, TargetMachineFactoryFn},
    },
    traits::ModuleBufferMethods,
};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::{DiagCtxt, DiagCtxtHandle, FatalError};
use rustc_middle::dep_graph::WorkProduct;
use tracing::{debug, trace};

use crate::NvvmCodegenBackend;
use crate::common::AsCCharPtr;
use crate::{LlvmMod, llvm};

pub struct ModuleBuffer(&'static mut llvm::ModuleBuffer);

unsafe impl Send for ModuleBuffer {}
unsafe impl Sync for ModuleBuffer {}

impl ModuleBuffer {
    pub(crate) fn new(m: &llvm::Module, is_thin: bool) -> ModuleBuffer {
        if is_thin {
            trace!(
                "serializing thin-LTO input via full-module bitcode; ThinLTO-specific shim APIs remain unwired on the Rust side"
            );
        }
        ModuleBuffer(unsafe { llvm::LLVMRustModuleBufferCreate(m) })
    }
}

impl ModuleBufferMethods for ModuleBuffer {
    fn data(&self) -> &[u8] {
        unsafe {
            trace!("Retrieving data in module buffer");
            let ptr = llvm::LLVMRustModuleBufferPtr(self.0);
            let len = llvm::LLVMRustModuleBufferLen(self.0);
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

impl Drop for ModuleBuffer {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustModuleBufferFree(&mut *(self.0 as *mut _));
        }
    }
}

#[allow(dead_code)]
pub struct ThinData(&'static mut llvm::ThinLTOData);

unsafe impl Send for ThinData {}
unsafe impl Sync for ThinData {}

impl Drop for ThinData {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMRustFreeThinLTOData(&mut *(self.0 as *mut _));
        }
    }
}

pub(crate) fn run_thin(
    _cgcx: &CodegenContext,
    modules: Vec<(String, ModuleBuffer)>,
    cached_modules: Vec<(SerializedModule<ModuleBuffer>, WorkProduct)>,
) -> (Vec<ThinModule<NvvmCodegenBackend>>, Vec<WorkProduct>) {
    debug!("Running thin LTO");
    let mut thin_buffers: Vec<ModuleBuffer> =
        Vec::with_capacity(modules.len() + cached_modules.len());
    let mut all_modules: Vec<SerializedModule<ModuleBuffer>> =
        Vec::with_capacity(modules.len() + cached_modules.len());
    let mut module_names = Vec::with_capacity(modules.len() + cached_modules.len());

    for (name, buf) in modules {
        thin_buffers.push(buf);
        module_names.push(CString::new(name).unwrap());
    }

    for (sm, wp) in cached_modules {
        all_modules.push(sm);
        module_names.push(CString::new(wp.cgu_name).unwrap());
    }

    let shared = Arc::new(ThinShared {
        data: (),
        thin_buffers,
        serialized_modules: all_modules,
        module_names,
    });

    let mut opt_jobs = Vec::with_capacity(shared.module_names.len());
    for module_index in 0..shared.module_names.len() {
        opt_jobs.push(ThinModule {
            shared: shared.clone(),
            idx: module_index,
        });
    }

    (opt_jobs, vec![])
}

pub(crate) unsafe fn optimize_and_codegen_thin(
    cgcx: &CodegenContext,
    prof: &SelfProfilerRef,
    shared_emitter: &SharedEmitter,
    _tm_factory: TargetMachineFactoryFn<NvvmCodegenBackend>,
    thin_module: ThinModule<NvvmCodegenBackend>,
) -> CompiledModule {
    let module_name = &thin_module.shared.module_names[thin_module.idx];
    let dcx = DiagCtxt::new(Box::new(shared_emitter.clone()));
    let llcx = unsafe { llvm::LLVMRustContextCreate(cgcx.fewer_names) };
    let llmod = parse_module(
        llcx,
        module_name.to_str().unwrap(),
        thin_module.data(),
        dcx.handle(),
    )
    .unwrap_or_else(|err| err.raise()) as *const _;

    let module =
        ModuleCodegen::new_regular(thin_module.name().to_string(), LlvmMod { llcx, llmod });
    unsafe { crate::back::codegen(cgcx, prof, shared_emitter, module, &cgcx.module_config) }
        .unwrap_or_else(|err| err.raise())
}

pub(crate) fn parse_module<'a>(
    cx: &'a llvm::Context,
    name: &str,
    data: &[u8],
    dcx: DiagCtxtHandle<'_>,
) -> Result<&'a llvm::Module, FatalError> {
    unsafe {
        llvm::LLVMRustParseBitcodeForLTO(
            cx,
            data.as_ptr(),
            data.len(),
            name.as_c_char_ptr(),
            name.len(),
        )
        .ok_or_else(|| {
            let msg = "failed to parse bitcode for LTO module";
            crate::back::llvm_err(dcx, msg)
        })
    }
}

/// Thin-LTO "Green" cached modules come back as on-disk bitcode files rather than
/// in-memory `ModuleBuffer`s; load each into a `Mmap` and wrap it in the upstream
/// `SerializedModule::FromUncompressedFile` variant (Phase-3b Tier-3: replaces the
/// old `SerializedModule::from_file` helper that upstream deleted).
pub(crate) fn load_serialized_module_for_thin_lto(
    path: &std::path::Path,
    dcx: DiagCtxtHandle<'_>,
) -> SerializedModule<ModuleBuffer> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(err) => crate::back::llvm_err(
            dcx,
            &format!(
                "failed to open thin-LTO bitcode {}: {}",
                path.display(),
                err
            ),
        )
        .raise(),
    };
    let mmap = match unsafe { rustc_data_structures::memmap::Mmap::map(file) } {
        Ok(m) => m,
        Err(err) => crate::back::llvm_err(
            dcx,
            &format!(
                "failed to mmap thin-LTO bitcode {}: {}",
                path.display(),
                err
            ),
        )
        .raise(),
    };
    SerializedModule::FromUncompressedFile(mmap)
}
