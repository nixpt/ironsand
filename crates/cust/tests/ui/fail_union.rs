use cust_derive::KernelDescriptor;

#[derive(KernelDescriptor)]
#[kernel_name = "foo"]
union BadUnion {
    a: u32,
    b: f32,
}

fn main() {}
