use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::{BufReader, BufRead};
use std::path::Path;

// Common utility functions for PTB examples.

// Gathers the set of words from space-separated corpus.
pub fn make_vocab<P: AsRef<Path>>(filename: P) -> Result<HashMap<String, u32>, io::Error> {
    let reader = BufReader::new(File::open(filename.as_ref())?);
    let mut vocab = HashMap::<String, u32>::new();
    for line in reader.lines() {
        let l = format!("<s>{}<s>", line.unwrap());
        for word in l.split(" ") {
            if !vocab.contains_key(word) {
                let id = vocab.len() as u32;
                vocab.insert(word.to_string(), id);
            }
        }
    }
    Ok(vocab)
}

// Generates word ID list using corpus and vocab.
pub fn load_corpus<P: AsRef<Path>>(
    filename: P,
    vocab: &HashMap<String, u32>,
) -> Result<Vec<Vec<u32>>, io::Error> {
    let reader = BufReader::new(File::open(filename.as_ref())?);
    let mut corpus = vec![];
    for line in reader.lines() {
        let l = format!("<s>{}<s>", line.unwrap());
        corpus.push(l.split(" ").map(|word| vocab[word]).collect::<Vec<_>>());
    }
    Ok(corpus)
}

// Counts output labels in the corpus.
pub fn count_labels<Corpus, Sentence>(corpus: Corpus) -> usize
where
    Corpus: AsRef<[Sentence]>,
    Sentence: AsRef<[u32]>,
{
    corpus.as_ref().iter().fold(0, |sum, sent| {
        sum + sent.as_ref().len() - 1
    })
}

// Extracts a minibatch from loaded corpus
pub fn make_batch<Corpus, Sentence>(
    corpus: Corpus,
    sent_ids: &[usize],
    eos_id: u32,
) -> Vec<Vec<u32>>
where
    Corpus: AsRef<[Sentence]>,
    Sentence: AsRef<[u32]>,
{
    let corpus = corpus.as_ref();
    let batch_size = sent_ids.len();
    let max_len = sent_ids
        .iter()
        .map(|&sid| corpus[sid].as_ref().len())
        .max()
        .unwrap();
    let mut batch = vec![vec![eos_id; batch_size]; max_len];
    for i in 0..batch_size {
        let sent = corpus[sent_ids[i]].as_ref();
        for j in 0..sent.len() {
            batch[j][i] = sent[j];
        }
    }
    batch
}
