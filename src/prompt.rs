use crate::assistant::Assistant;
use anyhow::Result;
use std::cell::Cell;

pub trait Prompt {
    fn run(&self, assistant: &mut Assistant, context: Option<String>) -> Result<String>;
}

pub struct StringReplacer {
    key: String,
    context: String, // TODO: turn this into a getter closure
}

impl StringReplacer {
    pub fn new(key: String, context: String) -> Self {
        Self { key, context }
    }
}

impl Prompt for StringReplacer {
    fn run(&self, assistant: &mut Assistant, context: Option<String>) -> Result<String> {
        let ctx = context.unwrap_or_else(|| String::from(""));
        let _a = assistant;
        Ok(ctx.replace(&self.key, &self.context))
    }
}

pub struct SmartReplacer {
    key: String,
    prompt: String,
}

impl Prompt for SmartReplacer {
    fn run(&self, assistant: &mut Assistant, context: Option<String>) -> Result<String> {
        let ctx = context.unwrap_or_else(|| String::from(""));
        let answer = assistant.answer(&self.prompt)?;
        Ok(ctx.replace(&self.key, &answer))
    }
}

pub struct Memory {
    memory: Cell<String>,
}

pub struct Simple;

impl Prompt for Simple {
    fn run(&self, assistant: &mut Assistant, context: Option<String>) -> Result<String> {
        let ctx = context.unwrap_or_else(|| String::from(""));
        assistant.answer(&ctx)
    }
}

pub struct PromptChain {
    prompts: Vec<Box<dyn Prompt>>,
}

impl Prompt for PromptChain {
    fn run(&self, assistant: &mut Assistant, context: Option<String>) -> Result<String> {
        let acc = context.unwrap_or_else(|| String::from(""));
        Ok(self
            .prompts
            .iter()
            .try_fold(acc, |acc, prompt| prompt.run(assistant, Some(acc)))?)
    }
}
