extern crate primitiv;
extern crate rand;

use rand::{thread_rng, Rng};
use std::cmp::min;
use std::env::args;
use std::io::{stdin, stdout, BufRead, Write};
use std::process::exit;

use primitiv::Graph;
use primitiv::Model;
use primitiv::ModelImpl;
use primitiv::Node;
use primitiv::Optimizer;
use primitiv::Parameter;
use primitiv::devices as D;
use primitiv::initializers as I;
use primitiv::functions as F;
use primitiv::optimizers as O;

mod lstm;
mod utils;
use lstm::LSTM;
use utils::{count_labels, line_to_sent, load_corpus, load_ppl, make_batch, make_inv_vocab,
            make_vocab, save_ppl};

const SRC_VOCAB_SIZE: usize = 4000;
const TRG_VOCAB_SIZE: usize = 5000;
const NUM_EMBED_UNITS: u32 = 512;
const NUM_HIDDEN_UNITS: u32 = 512;
const BATCH_SIZE: usize = 64;
const MAX_EPOCH: u32 = 100;
const DROPOUT_RATE: f32 = 0.5;
const GENERATION_LIMIT: usize = 32;

static SRC_TRAIN_FILE: &'static str = "data/train.en";
static TRG_TRAIN_FILE: &'static str = "data/train.ja";
static SRC_VALID_FILE: &'static str = "data/dev.en";
static TRG_VALID_FILE: &'static str = "data/dev.ja";

/// Encoder-decoder translation model.
pub struct EncoderDecoder {
    model: Model,
    dropout_rate: f32,
    psrc_lookup: Parameter,
    ptrg_lookup: Parameter,
    pwhy: Parameter,
    pby: Parameter,
    src_lstm: LSTM,
    trg_lstm: LSTM,
    trg_lookup: Node,
    why: Node,
    by: Node,
}

impl EncoderDecoder {
    pub fn new() -> Self {
        let mut m = EncoderDecoder {
            model: Model::new(),
            dropout_rate: DROPOUT_RATE,
            psrc_lookup: Parameter::new(),
            ptrg_lookup: Parameter::new(),
            pwhy: Parameter::new(),
            pby: Parameter::new(),
            src_lstm: LSTM::new(),
            trg_lstm: LSTM::new(),
            trg_lookup: Node::new(),
            why: Node::new(),
            by: Node::new(),
        };
        m.model.add_parameter("src_lookup", &mut m.psrc_lookup);
        m.model.add_parameter("trg_lookup", &mut m.ptrg_lookup);
        m.model.add_parameter("why", &mut m.pwhy);
        m.model.add_parameter("by", &mut m.pby);
        m.model.add_submodel("src_lstm", &mut m.src_lstm);
        m.model.add_submodel("trg_lstm", &mut m.trg_lstm);
        m
    }

    /// Initializes the model.
    pub fn init(
        &mut self,
        src_vocab_size: usize,
        trg_vocab_size: usize,
        embed_size: u32,
        hidden_size: u32,
    ) {
        self.psrc_lookup.init_by_initializer(
            [embed_size, src_vocab_size as u32],
            &I::XavierUniform::new(1.0),
        );
        self.ptrg_lookup.init_by_initializer(
            [embed_size, trg_vocab_size as u32],
            &I::XavierUniform::new(1.0),
        );
        self.pwhy.init_by_initializer(
            [trg_vocab_size as u32, hidden_size],
            &I::XavierUniform::new(1.0),
        );
        self.pby.init_by_initializer(
            [trg_vocab_size as u32],
            &I::Constant::new(1.0),
        );
        self.src_lstm.init(embed_size, hidden_size);
        self.trg_lstm.init(embed_size, hidden_size);
    }

    /// Encodes source sentences and prepare internal states.
    pub fn encode<Batch, Words>(&mut self, inputs: Batch, train: bool)
    where
        Batch: AsRef<[Words]>,
        Words: AsRef<[u32]>,
    {
        // Reversed encoding.
        let src_lookup = F::parameter::<Node>(&mut self.psrc_lookup);
        self.src_lstm.restart(None, None);
        for it in inputs.as_ref().iter().rev() {
            let x = F::pick(&src_lookup, it.as_ref(), 1);
            let x = F::dropout(x, self.dropout_rate, train);
            self.src_lstm.forward(&x);
        }

        // Initializes decoder states.
        self.trg_lookup = F::parameter(&mut self.ptrg_lookup);
        self.why = F::parameter(&mut self.pwhy);
        self.by = F::parameter(&mut self.pby);
        self.trg_lstm.restart(
            Some(self.src_lstm.get_c()),
            Some(self.src_lstm.get_h()),
        );
    }

    /// One step decoding.
    pub fn decode_step<Words: AsRef<[u32]>>(&mut self, trg_words: Words, train: bool) -> Node {
        let x = F::pick(&self.trg_lookup, trg_words.as_ref(), 1);
        let x = F::dropout(x, self.dropout_rate, train);
        let h = self.trg_lstm.forward(x);
        let h = F::dropout(h, self.dropout_rate, train);
        F::matmul(&self.why, h) + &self.by
    }

    /// Calculates the loss function over given target sentences.
    pub fn loss<Batch, Words>(&mut self, trg_batch: Batch, train: bool) -> Node
    where
        Batch: AsRef<[Words]>,
        Words: AsRef<[u32]>,
    {
        let trg_batch = trg_batch.as_ref();
        let mut losses = vec![];
        for i in 0..trg_batch.len() - 1 {
            let y = self.decode_step(&trg_batch[i], train);
            losses.push(F::softmax_cross_entropy_with_ids(
                y,
                &trg_batch[i + 1].as_ref(),
                0,
            ));
        }
        F::batch::mean(F::sum_vars(&losses))
    }
}

impl AsRef<Model> for EncoderDecoder {
    #[inline]
    fn as_ref(&self) -> &Model {
        &self.model
    }
}

impl AsMut<Model> for EncoderDecoder {
    #[inline]
    fn as_mut(&mut self) -> &mut Model {
        &mut self.model
    }
}

/// Training encoder decoder model.
pub fn train<O: Optimizer>(
    encdec: &mut EncoderDecoder,
    optimizer: &mut O,
    prefix: &str,
    mut best_valid_ppl: f32,
) {
    // Registers all parameters to the optimizer.
    optimizer.add_model(encdec);

    // Loads vocab.
    let src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE).unwrap();
    let trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE).unwrap();
    println!("#src_vocab: {}", src_vocab.len()); // == SRC_VOCAB_SIZE
    println!("#trg_vocab: {}", trg_vocab.len()); // == TRG_VOCAB_SIZE

    // Loads all corpus.
    let train_src_corpus = load_corpus(SRC_TRAIN_FILE, &src_vocab).unwrap();
    let train_trg_corpus = load_corpus(TRG_TRAIN_FILE, &trg_vocab).unwrap();
    let valid_src_corpus = load_corpus(SRC_VALID_FILE, &src_vocab).unwrap();
    let valid_trg_corpus = load_corpus(TRG_VALID_FILE, &trg_vocab).unwrap();
    let num_train_sents = train_trg_corpus.len();
    let num_valid_sents = valid_trg_corpus.len();
    let num_train_labels = count_labels(&train_trg_corpus);
    let num_valid_labels = count_labels(&valid_trg_corpus);
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

    // Batch randomizer.
    let mut rng = thread_rng();

    // Sentence IDs.
    let mut train_ids = (0..num_train_sents).collect::<Vec<_>>();
    let valid_ids = (0..num_valid_sents).collect::<Vec<_>>();

    // Computation graph.
    let mut g = Graph::new();
    Graph::set_default(&mut g);

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
            let src_batch = make_batch(&train_src_corpus, &batch_ids, &src_vocab);
            let trg_batch = make_batch(&train_trg_corpus, &batch_ids, &trg_vocab);

            g.clear();
            encdec.encode(src_batch, true);
            let loss = encdec.loss(trg_batch, true);
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
            let src_batch = make_batch(&valid_src_corpus, &batch_ids, &src_vocab);
            let trg_batch = make_batch(&valid_trg_corpus, &batch_ids, &trg_vocab);

            g.clear();
            encdec.encode(src_batch, false);
            let loss = encdec.loss(trg_batch, false);
            valid_loss += loss.to_float() * batch_ids.len() as f32;

            print!("{}\r", ofs);
            stdout().flush().unwrap();
            ofs += BATCH_SIZE;
        }

        let valid_ppl = (valid_loss / num_valid_labels as f32).exp();
        println!("  valid ppl = {}", valid_ppl);

        // Saves best model/optimizer.
        if valid_ppl < best_valid_ppl {
            best_valid_ppl = valid_ppl;
            print!("  saving model/optimizer ... ");
            stdout().flush().unwrap();
            encdec.save(format!("{}.model", prefix), true).unwrap();
            optimizer.save(format!("{}.optimizer", prefix)).unwrap();
            save_ppl(format!("{}.valid_ppl", prefix), best_valid_ppl).unwrap();
            println!("done.");
        }
    }
}

/// Generates translation by consuming stdin.
pub fn test(encdec: &mut EncoderDecoder) {
    let src_vocab = make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE).unwrap();
    let trg_vocab = make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE).unwrap();
    let inv_trg_vocab = make_inv_vocab(&trg_vocab);

    let mut g = Graph::new();
    Graph::set_default(&mut g);

    let stdin = stdin();
    for line in stdin.lock().lines() {
        let src_corpus = [line_to_sent(&line.unwrap(), &src_vocab)];
        let src_batch = make_batch(&src_corpus, &[0], &src_vocab);
        g.clear();
        encdec.encode(&src_batch, false);

        // Generates target words one-by-one.
        let mut trg_ids = vec![trg_vocab["<bos>"]];
        let eos_id = trg_vocab["<eos>"];
        while *trg_ids.last().unwrap() != eos_id {
            if trg_ids.len() > GENERATION_LIMIT + 1 {
                eprintln!(
                    "Warning: Sentence generation did not finish in {} iterations.",
                    GENERATION_LIMIT
                );
                trg_ids.push(eos_id);
                break;
            }
            let y = encdec.decode_step(&[*trg_ids.last().unwrap()], false);
            trg_ids.push(y.argmax(0)[0]);
        }

        // Prints the result.
        for i in 1..trg_ids.len() - 1 {
            if i > 1 {
                print!(" ");
            }
            print!("{}", inv_trg_vocab[trg_ids[i] as usize]);
        }
        println!();
    }
}

fn main() {
    let args: Vec<String> = args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} (train|resume|test) <model_prefix>", &args[0]);
        exit(1);
    }

    let mode = &args[1];
    let prefix = &args[2];
    eprintln!("mode: {}", mode);
    eprintln!("prefix: {}", prefix);
    if mode != "train" && mode != "resume" && mode != "test" {
        eprintln!("unknown mode: {}", mode);
        exit(1);
    }

    eprintln!("initializing device ... ");
    let mut dev = D::Naive::new(); // let mut dev = D::CUDA::new(0);
    D::set_default(&mut dev);
    eprintln!("done.");

    if mode == "train" {
        let mut encdec = EncoderDecoder::new();
        encdec.init(
            SRC_VOCAB_SIZE,
            TRG_VOCAB_SIZE,
            NUM_EMBED_UNITS,
            NUM_HIDDEN_UNITS,
        );
        let mut optimizer = O::Adam::default();
        optimizer.set_weight_decay(1e-6);
        optimizer.set_gradient_clipping(5.0);
        train(&mut encdec, &mut optimizer, prefix, 1e10);
    } else if mode == "resume" {
        eprint!("loading model/optimizer ... ");
        stdout().flush().unwrap();
        let mut encdec = EncoderDecoder::new();
        encdec.load(format!("{}.model", prefix), true).unwrap();
        let mut optimizer = O::Adam::default();
        optimizer.load(format!("{}.optimizer", prefix)).unwrap();
        let valid_ppl = load_ppl(format!("{}.valid_ppl", prefix)).unwrap();
        eprintln!("done.");
        train(&mut encdec, &mut optimizer, prefix, valid_ppl);
    } else {
        assert!(mode == "test");
        eprint!("loading model ... ");
        stdout().flush().unwrap();
        let mut encdec = EncoderDecoder::new();
        encdec.load(format!("{}.model", prefix), true).unwrap();
        eprintln!("done.");
        test(&mut encdec);
    }
    exit(0);
}
