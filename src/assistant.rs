use std::time::Instant;

use crate::token_output_stream::TokenOutputStream;
use anyhow::anyhow;
use anyhow::Result;
use async_stream::stream;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use futures_util::Stream;
use tokenizers::Tokenizer;

use crate::utils;

#[derive(Clone)]
pub struct Assistant {
    model: ModelWeights,
    tokenizer: Tokenizer,
}

impl Assistant {
    pub fn new(model: ModelWeights, tokenizer: Tokenizer) -> Self {
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
            .get("</s>")
            .ok_or_else(|| anyhow!("no end of text token?"))?;
        let device = utils::device()?;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.8), None);
        let mut generated_tokens = 0;
        // todo, we might need this to be not here dunno
        let mut assistant = self.clone();
        let start = Instant::now();
        let mut next_token = {
            let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = assistant.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        let prompt_len = tokens.len();
        let s = stream! {
            tokens.push(next_token.clone());
            if let Some(token) = tos.next_token(next_token)? {
                yield(Ok(token))
            }
            generated_tokens += 1;
            loop {
                let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                let logits = assistant.model.forward(&input, prompt_len + generated_tokens)?.squeeze(0)?;
                let start_at = tokens.len().saturating_sub(64);
                let logits = apply_repeat_penalty(
                    &logits,
                    1.1,
                    &tokens[start_at..],
                )?;
                next_token = logits_processor.sample(&logits)?;
                generated_tokens += 1;
                tokens.push(next_token.clone());
                if let Some(token) = tos.next_token(next_token)? {
                    yield Ok(token)
                }

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
        };

        Ok(s)
    }
}
