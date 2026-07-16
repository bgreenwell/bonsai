use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

mod bonsai_model {
    include!("../assets/tests/sklearn_onnx/regression_numeric/generated/model.rs");
}

fn bench_comparison(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("ONNX Comparison");

        // --- 1. BONSAI ---
        let model = bonsai_model::Model;
        let features = vec![0.5f32; 10];

        group.bench_function("bonsai_rust", |b| {
            b.iter(|| model.predict(black_box(&features)))
        });

        // --- 2. ONNX RUNTIME ---
        let onnx_path =
            Path::new("assets/tests/sklearn_onnx/regression_numeric/generated/model.onnx");
        if onnx_path.exists() {
            let mut session = Session::builder()
                .unwrap()
                .commit_from_file(onnx_path)
                .unwrap();
            let array = ndarray::Array2::from_shape_vec((1, 10), features.clone()).unwrap();

            group.bench_function("onnx_runtime_native", |b| {
                b.iter(|| {
                    let input_tensor = Tensor::from_array(array.clone()).unwrap();
                    let outputs = session.run(ort::inputs![input_tensor]).unwrap();
                    let res_data = outputs[0].try_extract_tensor::<f32>().unwrap().1;
                    let _res = res_data[0];
                    black_box(_res);
                })
            });
        }
        group.finish();
    }

    {
        // --- 1.5 OBLIVIOUS VS BRANCHY (BATCH) ---
        let mut ob_group = c.benchmark_group("Tree Structure Batch");
        let batch_size = 1024;
        let features = vec![0.5f32; 10 * batch_size];
        let t = vec![0.5f32; 10];
        let leaves = (0..64).map(|i| i as f32 * 0.1).collect::<Vec<_>>();

        ob_group.throughput(Throughput::Elements(batch_size as u64));

        #[inline(never)]
        fn predict_batch_oblivious(
            features: &[f32],
            thresholds: &[f32],
            leaves: &[f32],
            out: &mut [f32],
        ) {
            for (i, row) in features.chunks_exact(10).enumerate() {
                let index = (row[0] >= thresholds[0]) as usize
                    | ((row[1] >= thresholds[1]) as usize) << 1
                    | ((row[2] >= thresholds[2]) as usize) << 2
                    | ((row[3] >= thresholds[3]) as usize) << 3
                    | ((row[4] >= thresholds[4]) as usize) << 4
                    | ((row[5] >= thresholds[5]) as usize) << 5;
                out[i] = leaves[index];
            }
        }

        #[inline(never)]
        fn predict_batch_branchy(features: &[f32], thresholds: &[f32], out: &mut [f32]) {
            for (i, row) in features.chunks_exact(10).enumerate() {
                out[i] = if row[0] < thresholds[0] {
                    if row[1] < thresholds[1] {
                        if row[2] < thresholds[2] {
                            0.1
                        } else {
                            0.2
                        }
                    } else {
                        if row[3] < thresholds[3] {
                            0.3
                        } else {
                            0.4
                        }
                    }
                } else {
                    if row[4] < thresholds[4] {
                        if row[5] < thresholds[5] {
                            0.5
                        } else {
                            0.6
                        }
                    } else {
                        0.7
                    }
                };
            }
        }

        let mut out = vec![0.0f32; batch_size];

        ob_group.bench_function("batch_oblivious", |b| {
            b.iter(|| {
                predict_batch_oblivious(
                    black_box(&features),
                    black_box(&t),
                    black_box(&leaves),
                    black_box(&mut out),
                )
            })
        });

        ob_group.bench_function("batch_branchy", |b| {
            b.iter(|| {
                predict_batch_branchy(black_box(&features), black_box(&t), black_box(&mut out))
            })
        });

        ob_group.finish();
    }
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
