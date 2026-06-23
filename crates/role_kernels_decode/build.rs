use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("kernels")
        .copy_to("kernels.ptx")
        .build()
        .unwrap();
}
