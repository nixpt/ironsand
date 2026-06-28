use cust::kernel::KernelDescriptor; // trait
use cust_derive::KernelDescriptor; // derive macro

#[derive(KernelDescriptor)]
#[kernel_name = "scale"]
struct Scale(f32);

fn assert_kernel_descriptor<T: KernelDescriptor>() {}

fn main() {
    assert_kernel_descriptor::<Scale>();
    assert_eq!(<Scale as KernelDescriptor>::NAME, "scale");

    fn check_args(_: <Scale as KernelDescriptor>::Args) {}
    let _: fn((f32,)) = check_args;
}
