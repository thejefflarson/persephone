use anyhow::{anyhow, Result};
use candle_transformers::{
    models::quantized_mixformer::{Config, MixFormerSequentialForCausalLM},
    quantized_var_builder::VarBuilder,
};
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

const MODEL_REPO: &str = "lmz/candle-quantized-phi";
fn build_repo() -> Result<ApiRepo> {
    let api = Api::new()?;
    Ok(api.repo(Repo::with_revision(
        MODEL_REPO.into(),
        RepoType::Model,
        "main".into(),
    )))
}

const TOKENIZER: &str = "tokenizer.json";
#[derive(Debug)]
pub struct TokenizerFile(PathBuf);
impl TokenizerFile {
    pub fn download() -> Result<TokenizerFile> {
        let repo = build_repo()?;
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

const MODEL_FILE: &str = "model-v2-q4k.gguf";
#[derive(Debug)]
pub struct ModelFile(PathBuf);
impl ModelFile {
    pub fn download() -> Result<ModelFile> {
        let repo = build_repo()?;
        let filename = repo.get(MODEL_FILE)?;
        Ok(Self(filename))
    }

    pub fn model(&self) -> Result<MixFormerSequentialForCausalLM> {
        let vb = VarBuilder::from_gguf(&self.0, &device()?)?;
        MixFormerSequentialForCausalLM::new_v2(&Config::v2(), vb).map_err(|e| anyhow!(e))
    }
}

impl Display for ModelFile {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
