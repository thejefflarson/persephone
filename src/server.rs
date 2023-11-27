use anyhow::Result;
use async_graphql::{http::GraphiQLSource, EmptyMutation, Object, Schema, Subscription};
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
    async fn ask(&self) -> &'static str {
        "heyo"
    }
}

type AssistantSchema = Schema<QueryRoot, EmptyMutation, SubscriptionRoot>;

async fn graphiql() -> impl IntoResponse {
    Html(GraphiQLSource::build().finish())
}

async fn start() -> Result<()> {
    let schema = AssistantSchema::build(QueryRoot, EmptyMutation, SubscriptionRoot).finish();
    Ok(())
}
