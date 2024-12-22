use anyhow::{anyhow, Result};
use candle_core::DType::F16;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama, LlamaConfig};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
};
use tokenizers::Tokenizer;

use crate::utils::device;

fn build_repo(repo: &str) -> Result<ApiRepo> {
    let api = Api::new()?;
    Ok(api.repo(Repo::with_revision(
        repo.into(),
        RepoType::Model,
        "main".into(),
    )))
}

const TOKENIZER_REPO: &str = "HuggingFaceTB/SmolLM2-360M-Instruct";
const TOKENIZER: &str = "tokenizer.json";
#[derive(Debug)]
pub struct TokenizerFile(PathBuf);
impl TokenizerFile {
    pub fn download() -> Result<TokenizerFile> {
        let repo = build_repo(TOKENIZER_REPO)?;
        let filename = repo.get(TOKENIZER)?;
        Ok(Self(filename))
    }

    pub fn tokenizer(self) -> Result<Tokenizer> {
        Tokenizer::from_file(self.0).map_err(|e| anyhow!(e))
    }
}

impl Display for TokenizerFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

const MODEL_REPO: &str = "HuggingFaceTB/SmolLM2-360M-Instruct";
const MODEL_FILE: &str = "model.safetensors";
const CONFIG: &str = "config.json";
#[derive(Debug)]
pub struct ModelFile {
    config: PathBuf,
    filename: PathBuf,
}
impl ModelFile {
    pub fn download() -> Result<ModelFile> {
        let repo = build_repo(MODEL_REPO)?;
        let filename = repo.get(MODEL_FILE)?;
        let config = repo.get(CONFIG)?;
        Ok(Self { config, filename })
    }

    pub fn model(&self) -> Result<(Llama, Config)> {
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(&self.config)?)?;
        let config = config.into_config(false);
        let device = device()?;
        let filenames = vec![&self.filename];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, F16, &device)? };
        let llama = Llama::load(vb, &config)?;
        Ok((llama, config))
    }
}

impl Display for ModelFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?} {:?}", self.config, self.filename)
    }
}
