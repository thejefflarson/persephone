use std::time::Instant;

use anyhow::anyhow;
use anyhow::Result;
use candle_core::DType::F32;
use candle_core::Device::Cpu;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM;
use tokenizers::Tokenizer;
use tokio::sync::oneshot;

#[derive(Clone)]
pub struct Assistant {
    model: MixFormerSequentialForCausalLM,
    tokenizer: Tokenizer,
}

impl Assistant {
    pub fn new(model: MixFormerSequentialForCausalLM, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn answer(&mut self, prompt: &str) -> Result<String> {
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        let mut tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec(); // we reuse this as we go;
        let eos = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow!("no end of text token?"))?;
        let sample_len = 100;
        let device = Cpu;
        println!("running!");
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.9), None);
        let mut generated_tokens = 0;
        let start = Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let trimmed = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(trimmed, &device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?.squeeze(0)?.to_dtype(F32)?;
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            print!(
                "{}",
                self.tokenizer
                    .decode(&[next_token], true)
                    .map_err(|e| anyhow!(e))?
            );
            generated_tokens += 1;
            if next_token == eos {
                break;
            }
        }
        let result = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!(e))?;
        let done = start.elapsed();
        println!(
            "Generated {generated_tokens} tokens ({:.2} t/s)",
            generated_tokens as f64 / done.as_secs_f64()
        );
        Ok(String::from(result))
    }

    pub async fn answer_nonblocking(&self, prompt: &str) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        let mut copy = self.clone();
        let p = prompt.to_owned();
        rayon::spawn(move || {
            let result = copy.answer(&p);
            tx.send(result).unwrap()
        });
        rx.await?
    }
}
