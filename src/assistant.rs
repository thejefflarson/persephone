use std::time::Instant;

use crate::token_output_stream::TokenOutputStream;
use anyhow::anyhow;
use anyhow::Result;
use async_stream::stream;
use candle_core::DType;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::Llama;
use futures_util::Stream;
use tokenizers::Tokenizer;

use crate::utils;

pub struct Assistant {
    model: Llama,
    tokenizer: Tokenizer,
}

impl Assistant {
    pub fn new(model: Llama, tokenizer: Tokenizer) -> Self {
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
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let eos = *self
            .tokenizer
            .get_vocab(true)
            .get("</s>")
            .ok_or_else(|| anyhow!("no end of text token?"))?;
        let device = utils::device()?;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.8), None);
        let mut generated_tokens = 0usize;
        let mut index_pos = 0usize;
        let start = Instant::now();
        let s = stream! {
            loop {
                let (context_size, context_index) = if generated_tokens > 0 {
                    (1, index_pos)
                } else {
                    (tokens.len(), 0)
                };
                let ctx = &tokens[tokens.len().saturating_sub(context_size)..];
                let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;
                let logits = self.model.forward(&input, context_index)?.squeeze(0)?;
                let next_token = logits_processor.sample(&logits)?;
                generated_tokens += 1;
                index_pos += ctx.len();
                tokens.push(next_token);

                if next_token == eos {
                    break;
                }

                if let Some(text) = self.tokenizer.id_to_token(next_token) {
                    let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                    yield((Ok(text)))
                }
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
