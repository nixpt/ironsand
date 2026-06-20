# State

Updated: 2026-06-20T16:23:30-05:00

ironsand blessed end-to-end on this box (CUDA 13.3 /opt/cuda, LLVM 19.1.7 at /workspace/scratch/llvm19). Backend builds with --features llvm19; examples gemm/vecadd/matmul/async_api compile Rust->PTX and gemm+async_api run on the sm_120 5070 Ti. 3 commits on main (fork -> slim+attribution -> bless), no remote. Kept: codegen stack + cust host stack + blastoff (cuBLAS) + gpu_rand + examples + compiletests/xtask.
