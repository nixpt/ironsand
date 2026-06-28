use cust_derive::KernelDescriptor;

#[derive(KernelDescriptor)]
#[kernel_name = "foo"]
struct Generic<T>(T);

fn main() {}
