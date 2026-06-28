# Tips 

This section contains some tips on what to do and what not to do using the project.

## GPU kernels

- Generally don't derive `Debug` for structs in GPU crates. The codegen backend currently does not do much global
DCE (dead code elimination) so debug can really slow down compile times and make the PTX gigantic. This
will get much better in the future but currently it will cause some undesirable effects.

- Don't use recursion, CUDA allows it but threads have very limited stacks (local memory) and stack overflows
yield confusing `InvalidAddress` errors. If you are getting such an error, run the executable in cuda-memcheck,
it should yield a write failure to `Local` memory at an address of about 16mb. You can also put the PTX file through
`cuobjdump` and it should yield ptxas warnings for functions without a statically known stack usage.

- Prefer the typed [`Kernel`](../../kernel/struct.Kernel.html) API over the raw `launch!` macro. `Kernel<'a, Args>`
encodes the kernel's parameter signature in the type system, catching wrong argument counts and types at compile
time rather than at runtime. Use [`kernel_descriptor!`](typed_kernels.html) or `#[derive(KernelDescriptor)]`
to keep the host-side declaration in sync with the device-side `#[kernel]` function.
