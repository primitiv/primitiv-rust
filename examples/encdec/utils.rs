use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::exit;

/// Gathers the set of words from space-separated corpus and makes a vocabulary.
pub fn make_vocab<P: AsRef<Path>>(path: P, size: usize) -> Result<HashMap<String, u32>, io::Error> {
    if size < 3 {
        eprintln!("Vocab size should be <= 3.");
        exit(1);
    }
    let reader = io::BufReader::new(File::open(path.as_ref())?);

    // Counts all word existences.
    let mut freq = HashMap::<String, (u32, i32)>::new();
    reader.lines().for_each(|line| {
        line.unwrap().split(" ").for_each(|word| {
            let mut has_key = false;
            if let Some(value) = freq.get_mut(word) {
                (*value).0 += 1;
                has_key = true;
            }
            if !has_key {
                let index = freq.len() as i32;
                freq.insert(word.to_string(), (1, -index));
            }
        })
    });

    // Sorting.
    // Chooses top size-3 frequent words to make the vocabulary.
    let mut vocab = HashMap::<String, u32>::new();
    vocab.insert("<unk>".to_string(), 0);
    vocab.insert("<bos>".to_string(), 1);
    vocab.insert("<eos>".to_string(), 2);
    let mut freq = freq.into_iter().collect::<Vec<(String, (u32, i32))>>();
    freq.sort_by(|a, b| b.1.cmp(&a.1));
    freq.into_iter()
        .enumerate()
        .take_while(|&(i, _)| i < size - 3)
        .for_each(|(i, x)| {
            vocab.insert(x.0, i as u32 + 3);
        });
    Ok(vocab)
}

/// Generates ID-to-word dictionary.
pub fn make_inv_vocab(vocab: &HashMap<String, u32>) -> Vec<&str> {
    let mut vocab = vocab
        .iter()
        .map(|(s, i)| (*i, &s[..]))
        .collect::<Vec<(u32, &str)>>();
    vocab.sort_by(|a, b| a.0.cmp(&b.0));
    vocab.into_iter().map(|(_, s)| s).collect()
}

/// Generates word ID list from a sentence.
pub fn line_to_sent(line: &str, vocab: &HashMap<String, u32>) -> Vec<u32> {
    let unk_id = vocab["<unk>"];
    let covered = format!("<bos> {} <eos>", line);
    covered
        .split(" ")
        .map(|word| *vocab.get(word).unwrap_or(&unk_id))
        .collect()
}

/// Generates word ID list from a corpus.
/// All out-of-vocab words are replaced to <unk>.
pub fn load_corpus<P: AsRef<Path>>(
    path: P,
    vocab: &HashMap<String, u32>,
) -> Result<Vec<Vec<u32>>, io::Error> {
    let reader = io::BufReader::new(File::open(path.as_ref())?);
    let corpus = reader
        .lines()
        .map(|line| line_to_sent(&line.unwrap(), vocab))
        .collect();
    Ok(corpus)
}

/// Counts output labels in the corpus.
pub fn count_labels<Corpus, Sentence>(corpus: Corpus) -> usize
where
    Corpus: AsRef<[Sentence]>,
    Sentence: AsRef<[u32]>,
{
    corpus.as_ref().iter().fold(0, |sum, sent| {
        sum + sent.as_ref().len() - 1 // w/o <bos>
    })
}

/// Extracts a minibatch from loaded corpus
/// NOTE(chantera):
/// Lengths of all sentences are adjusted to the maximum one in the minibatch.
/// All additional subsequences are filled by <eos>. E.g.,
///   input: {
///     {<bos>, w1, <eos>},
///     {<bos>, w1, w2, w3, w4, <eos>},
///     {<bos>, w1, w2, <eos>},
///     {<bos>, w1, w2, w3, <eos>},
///   }
///   output: {
///     {<bos>, <bos>, <bos>, <bos>},
///     {   w1,    w1,    w1,    w1},
///     {<eos>,    w2,    w2,    w2},
///     {<eos>,    w3, <eos>,    w3},
///     {<eos>,    w4, <eos>, <eos>},
///     {<eos>, <eos>, <eos>, <eos>},
///   }
pub fn make_batch<Corpus, Sentence>(
    corpus: Corpus,
    sent_ids: &[usize],
    vocab: &HashMap<String, u32>,
) -> Vec<Vec<u32>>
where
    Corpus: AsRef<[Sentence]>,
    Sentence: AsRef<[u32]>,
{
    let corpus = corpus.as_ref();
    let batch_size = sent_ids.len();
    let eos_id = vocab["<eos>"];
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

/// Helper to save current ppl.
pub fn save_ppl<P: AsRef<Path>>(path: P, ppl: f32) -> Result<(), io::Error> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path.as_ref())?;
    writeln!(file, "{}", ppl)?;
    Ok(())
}

/// Helper to load last ppl.
pub fn load_ppl<P: AsRef<Path>>(path: P) -> Result<f32, io::Error> {
    let mut reader = io::BufReader::new(File::open(path.as_ref())?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(line.trim().parse::<f32>().unwrap())
}
