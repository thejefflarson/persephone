use crate::assistant::Assistant;
use anyhow::Result;
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use std::{cell::Cell, pin::pin};

#[async_trait]
pub trait BlockingPrompt {
    async fn run(&self, assistant: &Assistant, context: Option<String>) -> Result<String>;
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

#[async_trait]
impl BlockingPrompt for StringReplacer {
    async fn run(&self, assistant: &Assistant, context: Option<String>) -> Result<String> {
        let ctx = context.unwrap_or_else(|| String::from(""));
        let _a = assistant;
        Ok(ctx.replace(&self.key, &self.context))
    }
}

pub struct SmartReplacer {
    key: String,
    prompt: String,
}

impl SmartReplacer {
    pub fn new(key: String, prompt: String) -> Self {
        Self { key, prompt }
    }
}

#[async_trait]
impl BlockingPrompt for SmartReplacer {
    async fn run(&self, assistant: &Assistant, context: Option<String>) -> Result<String> {
        let ctx = context.unwrap_or_else(|| String::from(""));
        let mut answer = String::from("");
        let mut stream = pin!(assistant.answer(self.prompt.clone()).await?);
        while let Some(res) = stream.next().await {
            answer = res?;
        }
        Ok(ctx.replace(&self.key, &answer))
    }
}

pub struct Memory {
    memory: Cell<String>,
}

#[derive(Default)]
pub struct SimplePrompt;

impl SimplePrompt {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl BlockingPrompt for SimplePrompt {
    async fn run(&self, assistant: &Assistant, context: Option<String>) -> Result<String> {
        let ctx = context.unwrap_or_else(|| String::from(""));
        let mut stream = pin!(assistant.answer(ctx).await?);
        let mut answer = String::from("");
        while let Some(res) = stream.next().await {
            answer.push_str(&res?);
        }
        Ok(answer)
    }
}

pub struct SimpleStream;

impl SimpleStream {
    async fn run<'a>(
        &'a self,
        assistant: &'a Assistant,
        context: Option<String>,
    ) -> Result<impl Stream<Item = Result<String>> + 'a> {
        let ctx = context.unwrap_or_else(|| String::from("")).clone();
        assistant.answer(ctx).await
    }
}

async fn run_chain(
    prompts: Vec<impl BlockingPrompt>,
    assistant: &Assistant,
    context: Option<String>,
) -> Result<String> {
    let mut acc = context.unwrap_or_else(|| String::from(""));
    for prompt in prompts {
        acc = prompt.run(assistant, Some(acc)).await?
    }
    Ok(acc)
}
