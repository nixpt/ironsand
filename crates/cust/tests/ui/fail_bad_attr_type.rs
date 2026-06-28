use cust_derive::KernelDescriptor;

#[derive(KernelDescriptor)]
#[kernel_name = 42]
struct BadAttrType(f32);

fn main() {}
