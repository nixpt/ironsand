use cust::kernel::KernelDescriptor; // trait
use cust_derive::KernelDescriptor;  // derive macro

#[derive(KernelDescriptor)]
#[kernel_name = "vecadd"]
struct VecAdd(f32, usize, f32, usize, f32);

fn assert_kernel_descriptor<T: KernelDescriptor>() {}

fn main() {
    assert_kernel_descriptor::<VecAdd>();

    // Verify the associated constants are correct.
    assert_eq!(<VecAdd as KernelDescriptor>::NAME, "vecadd");

    // Verify Args tuple matches the struct fields.
    fn check_args(_: <VecAdd as KernelDescriptor>::Args) {}
    let _: fn((f32, usize, f32, usize, f32)) = check_args;
}
