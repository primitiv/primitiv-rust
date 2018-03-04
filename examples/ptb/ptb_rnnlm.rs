extern crate primitiv;
extern crate rand;

use rand::{thread_rng, Rng};
use std::cmp::min;
use std::io::{stdout, Write};

use primitiv::Graph;
use primitiv::Model;
use primitiv::Node;
use primitiv::Optimizer;
use primitiv::Parameter;
use primitiv::Shape;
use primitiv::devices as D;
use primitiv::functions as F;
use primitiv::initializers as I;
use primitiv::optimizers as O;

mod utils;

const NUM_HIDDEN_UNITS: u32 = 256;
const BATCH_SIZE: usize = 64;
const MAX_EPOCH: u32 = 100;

pub struct RNNLM {
    model: Model,
    pwlookup: Parameter,
    pwxs: Parameter,
    pwsy: Parameter,
}

impl RNNLM {
    pub fn new(vocab_size: usize) -> Self {
        let mut m = RNNLM {
            model: Model::new(),
            pwlookup: Parameter::from_initializer(
                [NUM_HIDDEN_UNITS, vocab_size as u32],
                &I::XavierUniform::new(1.0),
            ),
            pwxs: Parameter::from_initializer(
                [NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS],
                &I::XavierUniform::new(1.0),
            ),
            pwsy: Parameter::from_initializer(
                [vocab_size as u32, NUM_HIDDEN_UNITS],
                &I::XavierUniform::new(1.0),
            ),
        };
        m.model.add_parameter("pwlookup", &mut m.pwlookup);
        m.model.add_parameter("pwxs", &mut m.pwxs);
        m.model.add_parameter("pwsy", &mut m.pwsy);
        m
    }

    /// Forward function of RNNLM. Input data should be arranged below:
    /// inputs = {
    ///   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<s>)
    ///   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
    ///   ...,
    ///   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<s>)
    /// };
    pub fn forward<Batch, Sentence>(&mut self, inputs: Batch) -> Vec<Node>
    where
        Batch: AsRef<[Sentence]>,
        Sentence: AsRef<[u32]>,
    {
        let inputs = inputs.as_ref();
        let batch_size = inputs[0].as_ref().len() as u32;
        let wlookup = F::parameter(&mut self.pwlookup);
        let wxs = F::parameter(&mut self.pwxs);
        let wsy = F::parameter(&mut self.pwsy);
        let mut s = F::zeros(Shape::from_dims(&[NUM_HIDDEN_UNITS], batch_size));
        let mut outputs = vec![];
        for i in 0..inputs.len() - 1 {
            let w = F::pick(&wlookup, inputs[i].as_ref(), 1);
            let x = w + s;
            s = F::sigmoid(F::matmul(&wxs, &x));
            outputs.push(F::matmul(&wsy, &s));
        }
        outputs
    }

    /// Loss function.
    pub fn forward_loss<Batch, Sentence>(&self, outputs: &[Node], inputs: Batch) -> Node
    where
        Batch: AsRef<[Sentence]>,
        Sentence: AsRef<[u32]>,
    {
        let mut losses = vec![];
        for i in 0..outputs.len() {
            losses.push(F::softmax_cross_entropy_with_ids(
                &outputs[i],
                inputs.as_ref()[i + 1].as_ref(),
                0,
            ));
        }
        F::batch::mean(F::sum_nodes(&losses))
    }
}

impl AsRef<Model> for RNNLM {
    #[inline]
    fn as_ref(&self) -> &Model {
        &self.model
    }
}

impl AsMut<Model> for RNNLM {
    #[inline]
    fn as_mut(&mut self) -> &mut Model {
        &mut self.model
    }
}

fn main() {
    // Loads vocab.
    let vocab = utils::make_vocab("data/ptb.train.txt");
    println!("#vocab: {}", vocab.len()); // maybe 10000
    let eos_id = vocab["<s>"];

    // Loads all corpus.
    let train_corpus = utils::load_corpus("data/ptb.train.txt", &vocab);
    let valid_corpus = utils::load_corpus("data/ptb.valid.txt", &vocab);
    let num_train_sents = train_corpus.len();
    let num_valid_sents = valid_corpus.len();
    let num_train_labels = utils::count_labels(&train_corpus);
    let num_valid_labels = utils::count_labels(&valid_corpus);
    println!(
        "train: {} sentences, {} labels",
        num_train_sents,
        num_train_labels
    );
    println!(
        "valid: {} sentences, {} labels",
        num_valid_sents,
        num_valid_labels
    );

    let mut dev = D::Naive::new(); // let mut dev = D::CUDA::new(0);
    D::set_default(&mut dev);
    let mut g = Graph::new();
    Graph::set_default(&mut g);

    // Our LM.
    let mut lm = RNNLM::new(vocab.len());

    // Optimizer.
    let mut optimizer = O::Adam::default();
    optimizer.set_weight_decay(1e-6);
    optimizer.set_gradient_clipping(5.0);
    optimizer.add_model(&mut lm);

    // Batch randomizer.
    let mut rng = thread_rng();

    // Sentence IDs.
    let mut train_ids = (0..num_train_sents).collect::<Vec<_>>();
    let valid_ids = (0..num_valid_sents).collect::<Vec<_>>();

    // Train/valid loop.
    for epoch in 0..MAX_EPOCH {
        println!("epoch {}/{}:", epoch + 1, MAX_EPOCH);
        // Shuffles train sentence IDs.
        rng.shuffle(&mut train_ids);

        // Training.
        let mut train_loss = 0.0;
        let mut ofs = 0;
        while ofs < num_train_sents {
            let batch_ids = &train_ids[ofs..min(ofs + BATCH_SIZE, num_train_sents)];
            let batch = utils::make_batch(&train_corpus, &batch_ids, eos_id);
            g.clear();
            let outputs = lm.forward(&batch);
            let loss = lm.forward_loss(&outputs, batch);
            train_loss += loss.to_float() * batch_ids.len() as f32;

            optimizer.reset_gradients();
            loss.backward();
            optimizer.update();

            print!("{}\r", ofs);
            stdout().flush().unwrap();
            ofs += BATCH_SIZE;
        }

        let train_ppl = (train_loss / num_train_labels as f32).exp();
        println!("  train ppl = {}", train_ppl);

        // Validation.
        let mut valid_loss = 0.0;
        let mut ofs = 0;
        while ofs < num_valid_sents {
            let batch_ids = &valid_ids[ofs..min(ofs + BATCH_SIZE, num_valid_sents)];
            let batch = utils::make_batch(&valid_corpus, &batch_ids, eos_id);

            g.clear();
            let outputs = lm.forward(&batch);
            let loss = lm.forward_loss(&outputs, batch);
            valid_loss += loss.to_float() * batch_ids.len() as f32;

            print!("{}\r", ofs);
            stdout().flush().unwrap();
            ofs += BATCH_SIZE;
        }

        let valid_ppl = (valid_loss / num_valid_labels as f32).exp();
        println!("  valid ppl = {}", valid_ppl);
    }
}
