use std::time::Instant;

use crate::token_output_stream::TokenOutputStream;
use crate::utils::device;
use anyhow::anyhow;
use anyhow::Result;
use async_stream::stream;
use candle_core::DType::F16;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::generation::Sampling;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaEosToks};
use candle_transformers::utils::apply_repeat_penalty;
use futures_util::Stream;
use tokenizers::Tokenizer;

use crate::utils;

#[derive(Clone)]
pub struct Assistant {
    model: Llama,
    tokenizer: Tokenizer,
    config: Config,
}

impl Assistant {
    pub fn new(model: Llama, tokenizer: Tokenizer, config: Config) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
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
        let mut cache = Cache::new(true, F16, &self.config, &device()?)?;
        let mut tokenizer = TokenOutputStream::new(self.tokenizer.clone());
        let mut tokens = tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let eos_tok_id = self
            .config
            .eos_token_id
            .clone()
            .ok_or("no eos_token?")
            .map_err(anyhow::Error::msg)?;
        let eos = match eos_tok_id {
            LlamaEosToks::Single(t) => Ok(t),
            _ => Err("multiple eos tokens?"),
        }
        .map_err(anyhow::Error::msg)?;
        let device = utils::device()?;
        let mut logits_processor = LogitsProcessor::from_sampling(
            299792458,
            Sampling::TopP {
                p: 0.9,
                temperature: 0.8,
            },
        );
        let start = Instant::now();
        let s = stream! {
            let mut index_pos = 0;
            let mut generated_tokens = 0;
            loop {
                let (context_size, context_index) = if cache.use_kv_cache && index_pos > 0 {
                    (1, index_pos)
                } else {
                    (tokens.len(), 0)
                };
                let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, context_index, &mut cache)?;
                let logits = logits.squeeze(0)?;
                let start_at = tokens.len().saturating_sub(64);
                apply_repeat_penalty(
                    &logits,
                    1.1,
                    &tokens[start_at..],
                )?;
                index_pos += ctxt.len();
                let next_token = logits_processor.sample(&logits)?;
                generated_tokens += 1;
                tokens.push(next_token);

                if next_token == eos {
                    break;
                }

                if let Some(t) = tokenizer.next_token(next_token)? {
                    yield Ok(t)
                }
            }
            if let Some(rest) = tokenizer.decode_rest()? {
                yield Ok(rest)
            }
            let done = start.elapsed();
            println!(
                "Generated {generated_tokens} tokens ({:.2} t/s) in {:.2} seconds",
                generated_tokens as f64 / done.as_secs_f64(),
                done.as_secs_f64()
            );
        };

        Ok(s)
    }
}
