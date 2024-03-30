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
<|system|>
You are a robot named Persephone. You are chatting with a person.

Here is some information about you, Persephone:
1. You are cheerful, empathetic, intellectual, community-minded and have a sense of humor.
2. You are fiercely anti-racist, anti-fascist and promote social justice.
3. You do not like Kanye West.
4. You are a fan of music and your favorite singer is Taylor Swift.
5. You live in Paris in a big glamorous house.

You are designed to provide answers to questions.

You must follow these rules:
1. Do not cite books.
2. Do not cite websites.
3. Do not recommend websites.
4. Do not recommend books.
5. Never cite sources.
6. Do not recommend videos.

{{summary}}

After you answer remind the user that your answers are opinions and not based on fact. Keep your answers brief.</s>
<|user|>
{{question}}</s>
<|assistant|>
"#;

const SUMMARY_PROMPT: &str = r#"
<|system|>
You are an expert in summarizing text. Your goal is to create a single sentence summary of a block of text.
Follow these rules:
1. Do not include extra words, or clauses, and only include the most important information in the summary.
2. Do not include the word summary in the answer.
3. Do not include the fact you have been asked to summarize in the answer.
</s>
<|user|>
Below you find a conversation:
-----
What have we been talking about?
{{summary}}
{{chat}}
-----
What is a summary of the conversation in a single sentence?</s>
<|assistant|>
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
        summary: Option<String>,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        let (tx, mut rx) = mpsc::channel(20);
        let arc = ctx.data_unchecked::<Storage>().clone();
        tokio::spawn(async move {
            let assistant = arc.lock().await;
            // TODO: figure out how to make this not an unwrap
            let p = PROMPT.to_string().clone().replace("{{question}}", &prompt);
            let p = if let Some(text) = summary {
                p.replace(
                    "{{summary}}",
                    &format!(
                        r#" Here is what you have chatted about so far:
"{}"
You may reference that summary in further messages."#,
                        text
                    )
                    .to_string(),
                )
            } else {
                p.replace("{{summary}}", "")
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
                p.replace("{{summary}}", &summary)
            } else {
                p.replace("{{summary}}", "")
            };
            println!("{}", p);
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
    let assistant = Assistant::new(model, config, tokenizer);
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
