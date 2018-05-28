extern crate primitiv;
extern crate rand;

use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use primitiv::Graph;
use primitiv::Optimizer;
use primitiv::Parameter;

use primitiv::devices as D;
use primitiv::initializers as I;
use primitiv::node_functions as F;
use primitiv::optimizers as O;

const NUM_TRAIN_SAMPLES: u32 = 60000;
const NUM_TEST_SAMPLES: u32 = 10000;
const NUM_INPUT_UNITS: u32 = 28 * 28;
const NUM_HIDDEN_UNITS: u32 = 800;
const NUM_OUTPUT_UNITS: u32 = 10;
const BATCH_SIZE: u32 = 200;
const NUM_TRAIN_BATCHES: u32 = NUM_TRAIN_SAMPLES / BATCH_SIZE;
const NUM_TEST_BATCHES: u32 = NUM_TEST_SAMPLES / BATCH_SIZE;
const MAX_EPOCH: u32 = 100;

fn load_images<P: AsRef<Path>>(filename: P, n: u32) -> Vec<f32> {
    let mut reader = BufReader::new(File::open(filename.as_ref()).unwrap());
    reader.seek(SeekFrom::Start(16)).unwrap();
    let size = (n * NUM_INPUT_UNITS) as usize;
    let mut buf: Vec<u8> = Vec::with_capacity(size);
    reader.read_to_end(&mut buf).unwrap();
    let mut ret: Vec<f32> = Vec::with_capacity(size);
    for i in 0..size {
        ret.push(buf[i] as f32 / 255.0);
    }
    ret
}

fn load_labels<P: AsRef<Path>>(filename: P, n: u32) -> Vec<u8> {
    let mut reader = BufReader::new(File::open(filename.as_ref()).unwrap());
    reader.seek(SeekFrom::Start(8)).unwrap();
    let mut ret: Vec<u8> = Vec::with_capacity(n as usize);
    reader.read_to_end(&mut ret).unwrap();
    ret
}

fn main() {
    let train_inputs = load_images("data/train-images-idx3-ubyte", NUM_TRAIN_SAMPLES);
    let train_labels = load_labels("data/train-labels-idx1-ubyte", NUM_TRAIN_SAMPLES);
    let test_inputs = load_images("data/t10k-images-idx3-ubyte", NUM_TEST_SAMPLES);
    let test_labels = load_labels("data/t10k-labels-idx1-ubyte", NUM_TEST_SAMPLES);

    let mut dev = D::Naive::new(); // let mut dev = D::CUDA::new(0);
    D::set_default(&mut dev);

    let mut pw1 = Parameter::from_initializer([NUM_HIDDEN_UNITS, NUM_INPUT_UNITS],
                                              &I::XavierUniform::new(1.0));
    let mut pb1 = Parameter::from_initializer([NUM_HIDDEN_UNITS],
                                              &I::Constant::new(0.0));
    let mut pw2 = Parameter::from_initializer([NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS],
                                              &I::XavierUniform::new(1.0));
    let mut pb2 = Parameter::from_initializer([NUM_OUTPUT_UNITS],
                                              &I::Constant::new(0.0));

    let mut optimizer = O::SGD::new(0.5);
    optimizer.add_parameter(&mut pw1);
    optimizer.add_parameter(&mut pb1);
    optimizer.add_parameter(&mut pw2);
    optimizer.add_parameter(&mut pb2);

    let mut make_graph = |inputs: &[f32], train| {
        let x = F::input(([NUM_INPUT_UNITS], BATCH_SIZE), &inputs);
        let w1 = F::parameter(&mut pw1);
        let b1 = F::parameter(&mut pb1);
        let h = F::relu(F::matmul(w1, x) + b1);
        let h = F::dropout(h, 0.5, train);
        let w2 = F::parameter(&mut pw2);
        let b2 = F::parameter(&mut pb2);
        F::matmul(w2, h) + b2
    };

    let mut rng = thread_rng();
    let mut ids: Vec<usize> = (0usize..NUM_TRAIN_SAMPLES as usize).collect();

    let mut g = Graph::new();
    Graph::set_default(&mut g);

    for epoch in 0..MAX_EPOCH {
        rng.shuffle(&mut ids);

        for batch in 0..NUM_TRAIN_BATCHES {
            print!("\rTraining... {} / {}", batch + 1, NUM_TRAIN_BATCHES);
            let mut inputs: Vec<f32> = Vec::with_capacity((BATCH_SIZE * NUM_INPUT_UNITS) as usize);
            let mut labels: Vec<u32> = vec![0; BATCH_SIZE as usize];
            for i in 0..BATCH_SIZE {
                let id = ids[(i + batch * BATCH_SIZE) as usize];
                let from = id * NUM_INPUT_UNITS as usize;
                let to = (id + 1) * NUM_INPUT_UNITS as usize;
                inputs.extend_from_slice(&train_inputs[from..to]);
                labels[i as usize] = train_labels[id] as u32;
            }

            g.clear();

            let y = make_graph(&inputs, true);
            let loss = F::softmax_cross_entropy_with_ids(y, &labels, 0);
            let avg_loss = F::batch::mean(loss);

            optimizer.reset_gradients();
            avg_loss.backward();
            optimizer.update();
        }

        println!();

        let mut match_ = 0;

        for batch in 0..NUM_TEST_BATCHES {
            print!("\rTesting... {} / {}", batch + 1, NUM_TEST_BATCHES);
            let mut inputs: Vec<f32> = Vec::with_capacity((BATCH_SIZE * NUM_INPUT_UNITS) as usize);
            let from = (batch * BATCH_SIZE * NUM_INPUT_UNITS) as usize;
            let to = ((batch + 1) * BATCH_SIZE * NUM_INPUT_UNITS) as usize;
            inputs.extend_from_slice(&test_inputs[from..to]);

            g.clear();

            let y = make_graph(&inputs, false);

            let y_val = y.to_vector();
            for i in 0..BATCH_SIZE {
                let mut maxval = -1e10;
                let mut argmax: i32 = -1;
                for j in 0..NUM_OUTPUT_UNITS {
                    let v = y_val[(j + i * NUM_OUTPUT_UNITS) as usize];
                    if v > maxval {
                        maxval = v;
                        argmax = j as i32;
                    }
                }
                if argmax == test_labels[(i + batch * BATCH_SIZE) as usize] as i32 {
                    match_ += 1;
                }
            }
        }

        let accuracy = 100.0 * match_ as f32 / NUM_TEST_SAMPLES as f32;
        println!("\nepoch {}: accuracy: {:.2}%", epoch, accuracy);
    }
}
