use anyhow::{anyhow, Result};
use candle_core::DType::F16;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use std::{
    fmt::{Display, Formatter},
    fs::read,
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

const TOKENIZER_REPO: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
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

const MODEL_REPO: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const MODEL_FILE: &str = "model.safetensors";
const CONFIG_FILE: &str = "config.json";
#[derive(Debug)]
pub struct ModelFile {
    filename: PathBuf,
    config: PathBuf,
}
impl ModelFile {
    pub fn download() -> Result<ModelFile> {
        let repo = build_repo(MODEL_REPO)?;
        let filename = repo.get(MODEL_FILE)?;
        let config = repo.get(CONFIG_FILE)?;
        Ok(Self { filename, config })
    }

    pub fn model(&self) -> Result<Llama> {
        let config: LlamaConfig = serde_json::from_slice(&read(self.config.clone())?)?;
        let config = config.into_config(false);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vec![self.filename.clone()], F16, &device()?)?
        };
        let cache = Cache::new(true, F16, &config, &device()?).map_err(|e| anyhow!(e))?;
        Ok(Llama::load(vb, &cache, &config)?)
    }
}

impl Display for ModelFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.filename)
    }
}
