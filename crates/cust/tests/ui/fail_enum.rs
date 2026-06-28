use cust_derive::KernelDescriptor;

#[derive(KernelDescriptor)]
#[kernel_name = "foo"]
enum BadEnum {
    A,
}

fn main() {}
