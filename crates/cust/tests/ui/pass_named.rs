use cust::kernel::KernelDescriptor; // trait
use cust_derive::KernelDescriptor;  // derive macro

#[derive(KernelDescriptor)]
#[kernel_name = "saxpy"]
struct Saxpy {
    x: f32,
    y: f32,
    a: f32,
    n: usize,
}

fn assert_kernel_descriptor<T: KernelDescriptor>() {}

fn main() {
    assert_kernel_descriptor::<Saxpy>();
    assert_eq!(<Saxpy as KernelDescriptor>::NAME, "saxpy");

    fn check_args(_: <Saxpy as KernelDescriptor>::Args) {}
    let _: fn((f32, f32, f32, usize)) = check_args;
}
