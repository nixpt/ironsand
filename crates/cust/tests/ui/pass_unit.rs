use cust::kernel::KernelDescriptor; // trait
use cust_derive::KernelDescriptor; // derive macro

#[derive(KernelDescriptor)]
#[kernel_name = "noop"]
struct NoOp;

fn assert_kernel_descriptor<T: KernelDescriptor>() {}

fn main() {
    assert_kernel_descriptor::<NoOp>();
    assert_eq!(<NoOp as KernelDescriptor>::NAME, "noop");

    fn check_args(_: <NoOp as KernelDescriptor>::Args) {}
    let _: fn(()) = check_args;
}
