use anyhow::anyhow;
use anyhow::Result;
use candle_core::DType::F32;
use candle_core::Device;
use candle_core::Device::Cpu;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;

pub struct Assistant {
    model: ModelWeights,
    tokenizer: Tokenizer,
}

impl Assistant {
    pub fn new(model: ModelWeights, tokenizer: Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    pub fn answer(&mut self, prompt: &str) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec(); // we reuse this as we go;
        let prompt_len = tokens.len();
        let eos = *self
            .tokenizer
            .get_vocab(true)
            .get("</s>")
            .ok_or_else(|| anyhow!("no end of text token?"))?;
        let sample_len = 100;
        let device = Cpu;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(0.8), None);

        let mut next_token = {
            let input = Tensor::new(tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        for index in 0..sample_len {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, prompt_len + index)?.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?;
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
