use anyhow::{anyhow, Result};
use candle_core::quantized::gguf_file::Content;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use std::{
    fmt::{Display, Formatter},
    fs::File,
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

const MODEL_REPO: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
const MODEL_FILE: &str = "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf";
#[derive(Debug)]
pub struct ModelFile(PathBuf);
impl ModelFile {
    pub fn download() -> Result<ModelFile> {
        let repo = build_repo(MODEL_REPO)?;
        let filename = repo.get(MODEL_FILE)?;
        Ok(Self(filename))
    }

    pub fn model(&self) -> Result<ModelWeights> {
        let mut file = File::open(&self.0)?;
        let gguf = Content::read(&mut file)?;
        ModelWeights::from_gguf(gguf, &mut file, &device()?).map_err(|e| anyhow!(e))
    }
}

impl Display for ModelFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
