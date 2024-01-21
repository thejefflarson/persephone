use std::time::Instant;

use anyhow::anyhow;
use anyhow::Result;
use async_stream::stream;
use candle_core::DType::F32;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM;
use futures_util::Stream;
use tokenizers::Tokenizer;

use crate::utils;

#[derive(Clone)]
pub struct Assistant {
    model: MixFormerSequentialForCausalLM,
    tokenizer: Tokenizer,
}

impl Assistant {
    pub fn new(model: MixFormerSequentialForCausalLM, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub async fn answer<'a>(
        &'a self,
        prompt: String,
    ) -> Result<impl Stream<Item = Result<String>> + 'a> {
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
        let device = utils::device()?;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.9), None);
        let mut generated_tokens = 0;
        // todo, we might need this to be not here dunno
        let mut assistant = self.clone();
        let start = Instant::now();
        let s = stream! {
            loop {
                let context_size = if generated_tokens > 0 { 1 } else { tokens.len() };
                let trimmed = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = Tensor::new(trimmed, &device)?.unsqueeze(0)?;
                let logits = assistant.model.forward(&input)?.squeeze(0)?.to_dtype(F32)?;
                let next_token = logits_processor.sample(&logits)?;
                tokens.push(next_token);
                generated_tokens += 1;
                let decoded = self.tokenizer.decode(&[next_token], true).map_err(|e| anyhow!(e))?;
                yield (Ok(decoded));
                if next_token == eos {
                    break;
                }
            }
            let done = start.elapsed();
            println!(
                "Generated {generated_tokens} tokens ({:.2} t/s) in {:.2} seconds",
                generated_tokens as f64 / done.as_secs_f64(),
                done.as_secs_f64()
            );
            assistant.model.clear_kv_cache();
        };

        Ok(s)
    }
}
