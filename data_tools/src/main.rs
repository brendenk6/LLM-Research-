//! genesis-pack: Fast sequence packer for GENESIS training data.
//!
//! Reads cleaned JSONL files, tokenizes with tiktoken cl100k_base,
//! packs into fixed-length blocks, writes binary memmap + meta.json.
//!
//! Usage:
//!   cargo run --release -- --input-dir ../cleaned --output-dir ../packed --block-size 4096

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tiktoken_rs::cl100k_base;

/// GENESIS sequence packer — JSONL → tokenized binary blocks
#[derive(Parser, Debug)]
#[command(name = "genesis-pack")]
struct Args {
    /// Directory containing cleaned JSONL files
    #[arg(long, default_value = "../cleaned")]
    input_dir: PathBuf,

    /// Output directory for packed binary files
    #[arg(long, default_value = "../packed")]
    output_dir: PathBuf,

    /// Tokens per block (matches model max_seq_len)
    #[arg(long, default_value_t = 4096)]
    block_size: usize,

    /// Train/val split ratio (fraction for training)
    #[arg(long, default_value_t = 0.98)]
    train_ratio: f64,

    /// EOS token ID (tiktoken cl100k_base n_vocab = 100277, our EOS = 100279)
    #[arg(long, default_value_t = 100279)]
    eos_id: u32,

    /// Number of threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(Deserialize)]
struct JsonlRecord {
    text: Option<String>,
    content: Option<String>,
}

impl JsonlRecord {
    fn get_text(&self) -> Option<&str> {
        self.text
            .as_deref()
            .or(self.content.as_deref())
    }
}

fn main() {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }

    // Find all JSONL files
    let mut jsonl_files: Vec<PathBuf> = fs::read_dir(&args.input_dir)
        .expect("Cannot read input directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "jsonl")
                .unwrap_or(false)
        })
        .collect();
    jsonl_files.sort();

    if jsonl_files.is_empty() {
        eprintln!("No .jsonl files found in {:?}", args.input_dir);
        std::process::exit(1);
    }

    println!("Found {} JSONL files:", jsonl_files.len());
    for f in &jsonl_files {
        let size = fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        println!(
            "  {} ({:.1} GB)",
            f.file_name().unwrap().to_string_lossy(),
            size as f64 / 1e9
        );
    }

    // Initialize tokenizer
    println!("\nInitializing tiktoken cl100k_base...");
    let bpe = cl100k_base().expect("Failed to load cl100k_base");
    println!("Tokenizer ready (vocab: 100277 + 10 special = 100287)");

    // Phase 1: Tokenize all files in parallel, collect token streams
    println!("\n=== Phase 1: Tokenize ===");

    let total_tokens = AtomicU64::new(0);
    let total_docs = AtomicU64::new(0);
    let skipped_docs = AtomicU64::new(0);

    // Process each file, tokenize lines in parallel chunks
    let mut all_tokens: Vec<u32> = Vec::new();

    for jsonl_path in &jsonl_files {
        let fname = jsonl_path.file_name().unwrap().to_string_lossy();
        let file_size = fs::metadata(jsonl_path).map(|m| m.len()).unwrap_or(0);

        println!("\nTokenizing: {} ({:.1} GB)", fname, file_size as f64 / 1e9);

        let file = File::open(jsonl_path).expect("Cannot open file");
        let reader = BufReader::with_capacity(8 * 1024 * 1024, file);

        // Read lines in chunks for parallel tokenization
        let chunk_size = 10_000;
        let mut line_buffer: Vec<String> = Vec::with_capacity(chunk_size);

        let pb = ProgressBar::new(file_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );

        let mut bytes_read: u64 = 0;

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };
            bytes_read += line.len() as u64 + 1; // +1 for newline
            line_buffer.push(line);

            if line_buffer.len() >= chunk_size {
                let chunk_tokens: Vec<Vec<u32>> = line_buffer
                    .par_iter()
                    .filter_map(|line| {
                        let record: JsonlRecord = match serde_json::from_str(line) {
                            Ok(r) => r,
                            Err(_) => {
                                skipped_docs.fetch_add(1, Ordering::Relaxed);
                                return None;
                            }
                        };
                        let text = match record.get_text() {
                            Some(t) if t.len() >= 100 => t,
                            _ => {
                                skipped_docs.fetch_add(1, Ordering::Relaxed);
                                return None;
                            }
                        };
                        let tokens = bpe.encode_ordinary(text);
                        let token_ids: Vec<u32> =
                            tokens.into_iter().map(|t| t as u32).collect();
                        total_docs.fetch_add(1, Ordering::Relaxed);
                        total_tokens
                            .fetch_add(token_ids.len() as u64, Ordering::Relaxed);
                        Some(token_ids)
                    })
                    .collect();

                // Append to stream with EOS separators
                for doc_tokens in chunk_tokens {
                    all_tokens.extend_from_slice(&doc_tokens);
                    all_tokens.push(args.eos_id);
                }

                pb.set_position(bytes_read);
                line_buffer.clear();
            }
        }

        // Process remaining lines
        if !line_buffer.is_empty() {
            let chunk_tokens: Vec<Vec<u32>> = line_buffer
                .par_iter()
                .filter_map(|line| {
                    let record: JsonlRecord = match serde_json::from_str(line) {
                        Ok(r) => r,
                        Err(_) => {
                            skipped_docs.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };
                    let text = match record.get_text() {
                        Some(t) if t.len() >= 100 => t,
                        _ => {
                            skipped_docs.fetch_add(1, Ordering::Relaxed);
                            return None;
                        }
                    };
                    let tokens = bpe.encode_ordinary(text);
                    let token_ids: Vec<u32> =
                        tokens.into_iter().map(|t| t as u32).collect();
                    total_docs.fetch_add(1, Ordering::Relaxed);
                    total_tokens
                        .fetch_add(token_ids.len() as u64, Ordering::Relaxed);
                    Some(token_ids)
                })
                .collect();

            for doc_tokens in chunk_tokens {
                all_tokens.extend_from_slice(&doc_tokens);
                all_tokens.push(args.eos_id);
            }
        }

        pb.finish_with_message("done");
    }

    let n_tokens = total_tokens.load(Ordering::Relaxed);
    let n_docs = total_docs.load(Ordering::Relaxed);
    let n_skipped = skipped_docs.load(Ordering::Relaxed);

    println!("\n=== Tokenization Complete ===");
    println!("Documents: {} ({} skipped)", n_docs, n_skipped);
    println!(
        "Tokens: {} ({:.1}B)",
        n_tokens,
        n_tokens as f64 / 1e9
    );
    println!(
        "Token stream size: {:.1} GB",
        (all_tokens.len() * 4) as f64 / 1e9
    );

    // Phase 2: Pack into fixed blocks
    println!("\n=== Phase 2: Pack into {}-token blocks ===", args.block_size);

    let n_blocks = all_tokens.len() / args.block_size;
    // Trim to exact block boundary
    all_tokens.truncate(n_blocks * args.block_size);

    println!("Total blocks: {}", n_blocks);
    println!(
        "Packed size: {:.1} GB",
        (n_blocks * args.block_size * 4) as f64 / 1e9
    );

    // Split train/val
    let train_blocks = (n_blocks as f64 * args.train_ratio) as usize;
    let val_blocks = n_blocks - train_blocks;

    println!("Train blocks: {} ({:.1}%)", train_blocks, args.train_ratio * 100.0);
    println!("Val blocks: {} ({:.1}%)", val_blocks, (1.0 - args.train_ratio) * 100.0);

    // Phase 3: Write binary files
    println!("\n=== Phase 3: Write binary files ===");

    fs::create_dir_all(&args.output_dir).expect("Cannot create output directory");

    // Write train split
    let train_path = args.output_dir.join("train_input_ids.bin");
    let train_end = train_blocks * args.block_size;
    write_binary(&train_path, &all_tokens[..train_end]);
    println!(
        "Wrote: {} ({:.1} GB)",
        train_path.display(),
        (train_end * 4) as f64 / 1e9
    );

    // Write val split
    let val_path = args.output_dir.join("val_input_ids.bin");
    write_binary(&val_path, &all_tokens[train_end..]);
    println!(
        "Wrote: {} ({:.1} GB)",
        val_path.display(),
        ((n_blocks * args.block_size - train_end) * 4) as f64 / 1e9
    );

    // Write metadata
    let meta = serde_json::json!({
        "block_size": args.block_size,
        "dtype": "uint32",
        "vocab_size": 100287,
        "eos_id": args.eos_id,
        "train_blocks": train_blocks,
        "val_blocks": val_blocks,
        "total_tokens": n_tokens,
        "total_docs": n_docs,
        "skipped_docs": n_skipped,
        "source_files": jsonl_files.iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect::<Vec<_>>(),
    });

    let meta_path = args.output_dir.join("meta.json");
    let meta_file = File::create(&meta_path).expect("Cannot create meta.json");
    serde_json::to_writer_pretty(meta_file, &meta).expect("Cannot write meta.json");
    println!("Wrote: {}", meta_path.display());

    println!("\n=== Done ===");
    println!(
        "Total: {} blocks x {} tokens = {:.1}B tokens packed",
        n_blocks,
        args.block_size,
        (n_blocks * args.block_size) as f64 / 1e9
    );
}

fn write_binary(path: &PathBuf, tokens: &[u32]) {
    let file = File::create(path).expect("Cannot create binary file");
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);

    // Write as raw little-endian u32 bytes
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 4)
    };
    writer.write_all(bytes).expect("Write failed");
    writer.flush().expect("Flush failed");
}
