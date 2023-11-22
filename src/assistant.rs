use anyhow::anyhow;
use anyhow::Result;
use candle_core::DType::F32;
use candle_core::Device::Cpu;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM;
use tokenizers::Tokenizer;

pub struct Assistant {
    model: MixFormerSequentialForCausalLM,
    tokenizer: Tokenizer,
}

impl Assistant {
    pub fn new(model: MixFormerSequentialForCausalLM, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn answer(&mut self, prompt: &str) -> Result<String> {
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
        let mut logits_processor = LogitsProcessor::new(299792458, None, None);
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            // take a slice of where we have been, on the first loop it's the whole thing, and then
            // as we go it's the last item, because we're actually training the model to predict
            // each next word. Hilarious!
            let trimmed = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(trimmed, &device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?.squeeze(0)?.to_dtype(F32)?;
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if next_token == eos {
                break;
            }
        }
        let result = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!(e))?;
        Ok(String::from(result))
    }
}
