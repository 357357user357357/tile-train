// Phase-2 spike: prove a custom CubeCL kernel compiles and runs on CUDA.
// Vector addition on the GPU; result verified on CPU.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
fn vector_addition<F: Float>(lhs: &[F], rhs: &[F], out: &mut [F], dim: usize) {
    if ABSOLUTE_POS < dim {
        out[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
    }
}

fn main() {
    let client = CudaRuntime::client(&Default::default());
    let n = 3usize;

    let lhs = client.create_from_slice(f32::as_bytes(&[1.0f32, 2.0, 3.0]));
    let rhs = client.create_from_slice(f32::as_bytes(&[4.0f32, 5.0, 6.0]));
    let out = client.empty(n * core::mem::size_of::<f32>());

    vector_addition::launch::<f32>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(n as u32),
        // SAFETY: lengths match the buffers created above exactly.
        unsafe { BufferArg::from_raw_parts(lhs.clone(), n) },
        unsafe { BufferArg::from_raw_parts(rhs.clone(), n) },
        unsafe { BufferArg::from_raw_parts(out.clone(), n) },
        n,
    );

    let result = client.read_one(out).unwrap();
    let result = f32::from_bytes(&result);
    println!("result = {:?}", result);

    assert_eq!(result, &[5.0f32, 7.0, 9.0], "kernel output mismatch");
    println!("PHASE-2 SPIKE PASS: custom CubeCL kernel ran on CUDA and returned the right answer");
}
