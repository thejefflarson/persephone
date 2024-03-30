use anyhow::{anyhow, Result};
use candle_core::DType;
use candle_nn::VarBuilder;
use candle_transformers::models::mamba::{Config, Model};
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

fn build_repo(repo: &str, branch: &str) -> Result<ApiRepo> {
    let api = Api::new()?;
    Ok(api.repo(Repo::with_revision(
        repo.into(),
        RepoType::Model,
        branch.into(),
    )))
}

const TOKENIZER_REPO: &str = "EleutherAI/gpt-neox-20b";
const TOKENIZER: &str = "tokenizer.json";
#[derive(Debug)]
pub struct TokenizerFile(PathBuf);
impl TokenizerFile {
    pub fn download() -> Result<TokenizerFile> {
        let repo = build_repo(TOKENIZER_REPO, "main")?;
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

const MODEL_REPO: &str = "state-spaces/mamba-790m";
const MODEL_FILE: &str = "model.safetensors";
const CONFIG_FILE: &str = "config.json";
#[derive(Debug)]
pub struct ModelFile {
    model: PathBuf,
    config: PathBuf,
}
impl ModelFile {
    pub fn download() -> Result<ModelFile> {
        let repo = build_repo(MODEL_REPO, "refs/pr/1")?;
        let model = repo.get(MODEL_FILE)?;
        let config = repo.get(CONFIG_FILE)?;
        Ok(Self { model, config })
    }

    pub fn model(&self) -> Result<(Model, Config)> {
        let config: Config = serde_json::from_slice(&std::fs::read(self.config.clone())?)?;
        let device = device()?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&vec![self.model.clone()], DType::F32, &device)?
        };
        let model = Model::new(&config, vb.pp("backbone"))?;
        Ok((model, config))
    }
}

impl Display for ModelFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}, {:?}", self.model, self.config)
    }
}
