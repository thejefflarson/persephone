use std::time::Instant;

use crate::token_output_stream::TokenOutputStream;
use anyhow::anyhow;
use anyhow::Result;
use async_stream::stream;
use candle_core::DType;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mamba::{Config, Model, State};
use candle_transformers::utils::apply_repeat_penalty;
use futures_util::Stream;
use tokenizers::Tokenizer;

use crate::utils;

#[derive(Clone)]
pub struct Assistant {
    model: Model,
    config: Config,
    tokenizer: Tokenizer,
}

impl Assistant {
    pub fn new(model: Model, config: Config, tokenizer: Tokenizer) -> Self {
        Self {
            model,
            config,
            tokenizer,
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
        let mut tos = TokenOutputStream::new(self.tokenizer.clone());
        let mut tokens = tos
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let eos = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow!("no end of text token?"))?;
        let device = utils::device()?;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.8), None);
        let mut generated_tokens = 0;
        let start = Instant::now();
        let mut state = State::new(1, &self.config, &device)?;
        let mut next_logits = None;
        for &t in tokens.iter() {
            let input = Tensor::new(&[t], &device)?;
            let logits = self.model.forward(&input, &mut state)?;
            next_logits = Some(logits);
            if let Some(t) = tos.next_token(t)? {
                print!("{t}")
            }
        }
        let s = stream! {
            loop {
                if next_logits.is_none() {
                   yield Err(anyhow!("cannot work on an empty prompt"))
                }
                let logits = next_logits.as_ref().unwrap();
                let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
                let start_at = tokens.len().saturating_sub(64);
                let logits = apply_repeat_penalty(
                    &logits,
                    1.1,
                    &tokens[start_at..],
                )?;
                let next_token = logits_processor.sample(&logits)?;
                tokens.push(next_token.clone());

                generated_tokens += 1;
                if next_token == eos {
                    break;
                }

                if let Some(token) = tos.next_token(next_token)? {
                    print!("{token}");
                    yield Ok(token)
                }

                let input = Tensor::new(&[next_token], &device)?;
                next_logits = Some(self.model.forward(&input, &mut state)?);
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
