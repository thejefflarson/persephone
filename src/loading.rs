use anyhow::{anyhow, Result};
use candle_core::{DType, Device::Cpu};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
};
use tokenizers::Tokenizer;

const MODEL_REPO: &'static str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
fn build_repo() -> Result<ApiRepo> {
    let api = Api::new()?;
    Ok(api.repo(Repo::with_revision(
        MODEL_REPO.to_string(),
        RepoType::Model,
        "main".to_string(),
    )))
}

const TOKENIZER: &'static str = "tokenizer.json";
#[derive(Debug)]
pub struct TokenizerFile(PathBuf);
impl TokenizerFile {
    pub fn download() -> Result<TokenizerFile> {
        let repo = build_repo()?;
        let filename = repo.get(TOKENIZER)?;
        Ok(Self(filename))
    }

    pub fn tokenizer(self) -> Result<Tokenizer> {
        Ok(Tokenizer::from_file(self.0).map_err(|e| anyhow!(e))?)
    }
}

impl Display for TokenizerFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

const MODEL_FILE: &'static str = "model.safetensors";
#[derive(Debug)]
pub struct ModelFile(PathBuf);
impl ModelFile {
    pub fn download() -> Result<ModelFile> {
        let repo = build_repo()?;
        let filename = repo.get(MODEL_FILE)?;
        Ok(Self(filename))
    }

    pub fn model(&self, config: Config) -> Result<Llama> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vec![self.0.clone()], DType::BF16, &Cpu)?
        };
        let cache = Cache::new(true, DType::BF16, &config, &Cpu)?;
        let model = Llama::load(vb, &cache, &config)?;
        Ok(model)
    }
}

const CONFIG_FILE: &'static str = "config.json";
pub struct ConfigFile(PathBuf);
impl ConfigFile {
    pub fn download() -> Result<ConfigFile> {
        let repo = build_repo()?;
        let filename = repo.get(CONFIG_FILE)?;
        Ok(Self(filename))
    }

    pub fn config(&self) -> Result<Config> {
        let json = std::fs::read(&self.0)?;
        let config: LlamaConfig = serde_json::from_slice(&json)?;
        Ok(config.into_config(true))
    }
}

impl Display for ModelFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
