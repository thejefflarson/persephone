use std::sync::Arc;

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
use futures_util::{lock::Mutex, stream, Stream, StreamExt, TryStreamExt};
use std::pin::pin;
use tokio::{net::TcpListener, sync::mpsc};

struct QueryRoot;

#[Object]
impl QueryRoot {
    async fn ask(&self) -> &'static str {
        "heyo"
    }
}

type Storage = Arc<Mutex<Assistant>>;
struct SubscriptionRoot;

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

pub async fn start() -> Result<()> {
    let model = ModelFile::download()?.model()?;
    let tokenizer = TokenizerFile::download()?.tokenizer()?;
    let assistant = Assistant::new(model, tokenizer);
    let storage = Arc::new(Mutex::new(assistant));
    let schema = AssistantSchema::build(QueryRoot, EmptyMutation, SubscriptionRoot)
        .data(storage)
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
