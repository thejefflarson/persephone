use anyhow::Result;
use async_graphql::{
    futures_util::{stream, Stream},
    http::GraphiQLSource,
    EmptyMutation, Object, Schema, Subscription,
};
use axum::response::{Html, IntoResponse};

struct QueryRoot;

#[Object]
impl QueryRoot {
    async fn ask(&self) -> &'static str {
        "heyo"
    }
}

struct SubscriptionRoot;

#[Subscription]
impl SubscriptionRoot {
    async fn ask(&self) -> impl Stream<Item = &'static str> {
        stream::iter(vec!["heyo"])
    }
}

type AssistantSchema = Schema<QueryRoot, EmptyMutation, SubscriptionRoot>;

async fn graphiql() -> impl IntoResponse {
    Html(GraphiQLSource::build().finish())
}

async fn start() -> Result<()> {
    let _schema = AssistantSchema::build(QueryRoot, EmptyMutation, SubscriptionRoot).finish();
    Ok(())
}
