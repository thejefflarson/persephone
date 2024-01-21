use std::sync::Arc;

use crate::{
    assistant::{self, Assistant},
    loading::{ModelFile, TokenizerFile},
};

use async_graphql::{
    http::GraphiQLSource, Context, EmptyMutation, Error, GuardExt, Object, Result, Schema,
    Subscription,
};
use async_graphql_axum::{GraphQL, GraphQLSubscription};
use async_stream::stream;
use axum::{
    body::Body,
    extract::State,
    response::{
        sse::{Event, KeepAlive},
        Html, IntoResponse, Sse,
    },
    routing::{get, post},
    serve, Json, Router,
};
use futures_util::{lock::Mutex, Stream, StreamExt, TryStreamExt};
use serde::Deserialize;
use std::pin::pin;
use tokio::{net::TcpListener, sync::mpsc};

struct QueryRoot;

#[Object]
impl QueryRoot {
    async fn ask(&self, ctx: &Context<'_>, prompt: String) -> &'static str {
        "heyo"
    }
}

type Storage = Arc<Mutex<Assistant>>;
struct SubscriptionRoot;

// TODO: consider this for errors:
// https://github.com/tokio-rs/axum/blob/main/examples/anyhow-error-response/src/main.rs
#[Subscription]
impl SubscriptionRoot {
    async fn ask(
        &self,
        // Annoying but has to be second
        ctx: &Context<'_>,
        prompt: String,
    ) -> Result<impl Stream<Item = Result<String>> + '_> {
        let (tx, mut rx) = mpsc::channel(20);
        let arc = ctx.data_unchecked::<Storage>().clone();
        tokio::spawn(async move {
            let assistant = arc.lock().await;
            // TODO: figure out how to make this not an unwrap
            let tokens = assistant.answer(prompt).await.unwrap();
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

type AssistantSchema = Schema<QueryRoot, EmptyMutation, SubscriptionRoot>;

async fn graphiql() -> impl IntoResponse {
    Html(
        GraphiQLSource::build()
            .endpoint("/")
            .subscription_endpoint("/ws")
            .finish(),
    )
}

#[derive(Deserialize)]
struct StreamingArgs {
    pub prompt: String,
}

async fn streaming(State(storage): State<Storage>, Json(body): Json<StreamingArgs>) -> Body {
    let arc = storage.clone();
    let (tx, mut rx) = mpsc::channel(20);
    tokio::spawn(async move {
        let assistant = arc.lock().await;
        println!("{}", body.prompt);
        let tokens = assistant.answer(body.prompt).await.unwrap();
        let mut toks = pin!(tokens);
        while let Some(token) = toks.next().await {
            let tok = token
                .map(|it| it)
                .map_err(|e| axum::Error::new(e.to_string()));
            tx.send(tok).await.unwrap()
        }
    });

    let s = stream! {
        while let Some(token) = rx.recv().await {
            yield token
        }
    };
    Body::from_stream(s)
}

pub async fn start() -> Result<()> {
    let model = ModelFile::download()?.model()?;
    let tokenizer = TokenizerFile::download()?.tokenizer()?;
    let assistant = Assistant::new(model, tokenizer);
    let storage = Arc::new(Mutex::new(assistant));
    let schema = AssistantSchema::build(QueryRoot, EmptyMutation, SubscriptionRoot)
        .data(storage.clone())
        .finish();
    let app = Router::new()
        .route(
            "/",
            get(graphiql).post_service(GraphQL::new(schema.clone())),
        )
        .route("/streaming", post(streaming).with_state(storage))
        .route_service("/ws", GraphQLSubscription::new(schema));
    serve(TcpListener::bind("127.0.0.1:8000").await?, app).await?;
    Ok(())
}
