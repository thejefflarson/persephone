use std::{pin::pin, sync::Arc};

use crate::{
    assistant::Assistant,
    loading::{ModelFile, TokenizerFile},
};

use async_graphql::{
    http::GraphiQLSource, Context, EmptyMutation, Error, Object, Result, Schema, Subscription,
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

struct Query;

#[Object]
impl Query {
    async fn ask(&self, ctx: &Context<'_>, prompt: String) -> &'static str {
        "heyo"
    }
}

type Storage = Arc<Mutex<Assistant>>;
struct Subscription;

const PROMPT: &str = r#"
<|system|>
You are an AI assistant named Persephone. You are cheerful, empathetic, intellectual, community-minded and have a sense of humor. You are designed to provide answers to questions. Do not introduce yourself unless asked who you are by the user.

You must follow these rules:
1. Do not cite books.
2. Do not cite websites.
3. Do not recommend websites.
4. Do not recommend books.

From time to time, you should remind the user that your answers are opinions and not based on fact. Keep your answers brief.</s>
<|user|>
{{question}}</s>
<|assistant|>
"#;

// TODO: consider this for errors:
// https://github.com/tokio-rs/axum/blob/main/examples/anyhow-error-response/src/main.rs
#[Subscription]
impl Subscription {
    async fn ask(
        &self,
        // Annoying but has to be the second argument
        ctx: &Context<'_>,
        prompt: String,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        let (tx, mut rx) = mpsc::channel(20);
        let arc = ctx.data_unchecked::<Storage>().clone();
        tokio::spawn(async move {
            let assistant = arc.lock().await;
            // TODO: figure out how to make this not an unwrap
            let p = PROMPT.to_string().clone().replace("{{question}}", &prompt);
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
    let model = ModelFile::download()?.model()?;
    let tokenizer = TokenizerFile::download()?.tokenizer()?;
    let assistant = Assistant::new(model, tokenizer);
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
    serve(TcpListener::bind("127.0.0.1:8000").await?, app).await?;
    Ok(())
}
