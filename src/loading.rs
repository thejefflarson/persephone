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

const MODEL_REPO: &'static str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
fn build_repo(repo: Option<&str>) -> Result<ApiRepo> {
    let api = Api::new()?;
    Ok(api.repo(Repo::with_revision(
        repo.unwrap_or_else(|| MODEL_REPO).to_string(),
        RepoType::Model,
        "main".to_string(),
    )))
}

const TOKENIZER: &'static str = "tokenizer.json";
const TOKENIZER_REPO: &'static str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
#[derive(Debug)]
pub struct TokenizerFile(PathBuf);
impl TokenizerFile {
    pub fn download() -> Result<TokenizerFile> {
        let repo = build_repo(Some(TOKENIZER_REPO))?;
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

const MODEL_FILE: &'static str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
#[derive(Debug)]
pub struct ModelFile(PathBuf);
impl ModelFile {
    pub fn download() -> Result<ModelFile> {
        let repo = build_repo(None)?;
        let filename = repo.get(MODEL_FILE)?;
        Ok(Self(filename))
    }

    pub fn model(&self) -> Result<ModelWeights> {
        let mut file = File::open(self.0.clone())?;
        let gguf = Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(gguf, &mut file)?;
        Ok(model)
    }
}

impl Display for ModelFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
