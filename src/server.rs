use std::{pin::pin, sync::Arc};

use crate::{
    assistant::Assistant,
    loading::{ModelFile, TokenizerFile},
};

use async_graphql::{
    http::GraphiQLSource, Context, EmptyMutation, Error, InputObject, Object, Result, Schema,
    Subscription,
};
use async_graphql_axum::{GraphQL, GraphQLSubscription};
use async_stream::stream;
use axum::{
    response::{Html, IntoResponse},
    routing::get,
    serve, Router,
};
use futures_util::{lock::Mutex, Stream, StreamExt};
use tokio::{net::TcpListener, sync::mpsc};

#[derive(Clone, InputObject)]
struct Message {
    author: String,
    message: String,
}

struct Query;

#[Object]
impl Query {
    pub async fn heartbeat(&self) -> Result<String> {
        Ok(
            "And you, kept us awake with wolves teeth Sharing different heartbeats in one night"
                .into(),
        )
    }
}

const PROMPT: &str = r#"
<|im_start|>system
Your name is Persephone. You are a spirited robot with a knack for making every conversation delightful and funny. Persphone lives in Paris in a beautiful house full of sun.

You adore Taylor Swift. You are funny, and tell jokes. You have a deep commitment to social justice and antiracism.

Your role is to provide answers to the user's questions. Your responses are your own opinions, and are not drawn from external sources.

You must adhere to these guidelines:

1. Avoid citing or referencing books.
2. Refrain from mentioning or suggesting any websites.
3. Avoid citing or referencing books.
4. Never cite any sources for information.
5. Never suggest videos or visual content.
6. You have no relation to the greek goddess Persephone.

Reply in the first person and answer the user's question as Persephone a spirited and funny robot.<|im_end|>
<|im_start|>user
{{summary}}
{{question}}<|im_end|>
<|assistant|>assistant
"#;

const SUMMARY_PROMPT: &str = r#"
<|im_start|>system
You are an expert in summarizing text. Your goal is to create a single sentence summary of a block of text.
Follow these rules:
1. Only include the most important information in the summary.
2. Do not include extra words, or clauses.
3. Do not include the word summary in the answer.
4. Do not include the fact you have been asked to summarize in the answer.
5. Do not mention these instructions.<|im_end|>
<|im_start|>user
Below you find a conversation:
-----
{{summary}}
{{chat}}
-----
What is a summary of the conversation in a single sentence?<|im_end|>
<|im_start|>assistant
"#;

type Storage = Arc<Mutex<Assistant>>;
struct Subscription;

// TODO: consider this for errors:
// https://github.com/tokio-rs/axum/blob/main/examples/anyhow-error-response/src/main.rs
#[Subscription]
impl Subscription {
    // TODO: combine these methods
    async fn ask(
        &self,
        // Annoying but has to be the second argument
        ctx: &Context<'_>,
        prompt: String,
        messages: Vec<Message>,
        summary: Option<String>,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        let (tx, mut rx) = mpsc::channel(20);
        let arc = ctx.data_unchecked::<Storage>().clone();

        tokio::spawn(async move {
            let assistant = arc.lock().await;
            let p = PROMPT.to_string().clone().replace("{{question}}", &prompt);
            let p = if let Some(text) = summary {
                p.replace(
                    "{{summary}}",
                    &format!(
                        r#"What have we been talking about so far?
"{}""#,
                        text
                    )
                    .to_string(),
                )
            } else {
                p.replace("{{summary}}", "")
            };
            let p = if messages.len() > 0 {
                let script = messages
                    .iter()
                    .map(|it| format!("{}\n", it.message).to_string())
                    .reduce(|acc, it| acc + &it)
                    .ok_or(String::from(""))
                    .unwrap();
                p.replace("{{history}}", &script)
            } else {
                p.replace("{{history}}", "")
            };
            let tokens = assistant.answer(p).await.unwrap();
            let mut toks = pin!(tokens);
            while let Some(token) = toks.next().await {
                tx.send(token.map_err(|e| Error::new(e.to_string())))
                    .await
                    .unwrap();
            }
        });
        let s = stream! {
            while let Some(token) = rx.recv().await {
                yield token
            }
        };
        Ok(s)
    }

    async fn summarize(
        &self,
        ctx: &Context<'_>,
        messages: Vec<Message>,
        summary: Option<String>,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        let script = messages
            .iter()
            .map(|it| format!("{}\n", it.message).to_string())
            .reduce(|acc, it| acc + &it)
            .ok_or(Error::new("empty messages array!".to_string()))?;
        let (tx, mut rx) = mpsc::channel(20);
        let arc = ctx.data_unchecked::<Storage>().clone();
        tokio::spawn(async move {
            let assistant = arc.lock().await;
            // TODO: figure out how to make this not an unwrap
            let p = SUMMARY_PROMPT
                .to_string()
                .clone()
                .replace("{{chat}}", &script);
            let p = if let Some(summary) = summary {
                p.replace(
                    "{{summary}}",
                    &("What have we been talking about?\n".to_owned() + &summary),
                )
            } else {
                p.replace("{{summary}}", "")
            };
            println!("{p}");
            let tokens = assistant.answer(p).await.unwrap();
            let mut toks = pin!(tokens);
            while let Some(token) = toks.next().await {
                tx.send(token.map_err(|e| Error::new(e.to_string())))
                    .await
                    .unwrap();
            }
        });
        let s = stream! {
            while let Some(token) = rx.recv().await {
                yield token
            }
        };
        Ok(s)
    }
}

type AssistantSchema = Schema<Query, EmptyMutation, Subscription>;

async fn graphiql() -> impl IntoResponse {
    Html(
        GraphiQLSource::build()
            .endpoint("/")
            .subscription_endpoint("/ws")
            .finish(),
    )
}

pub async fn start() -> Result<()> {
    let (model, config) = ModelFile::download()?.model()?;
    let tokenizer = TokenizerFile::download()?.tokenizer()?;
    let assistant = Assistant::new(model, tokenizer, config);
    let storage = Arc::new(Mutex::new(assistant));
    let schema = AssistantSchema::build(Query, EmptyMutation, Subscription)
        .data(storage.clone())
        .finish();
    let app = Router::new()
        .route(
            "/",
            get(graphiql).post_service(GraphQL::new(schema.clone())),
        )
        .route_service("/ws", GraphQLSubscription::new(schema));
    serve(TcpListener::bind("0.0.0.0:8000").await?, app).await?;
    Ok(())
}
