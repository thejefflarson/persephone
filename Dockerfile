FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo chef cook --release --recipe-path recipe.json
# Build application
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release
RUN /app/target/release/persephone download

FROM rust:1
COPY --from=builder /app/target/release/persephone /usr/local/bin/personal-assistant
COPY --from=builder /root/.cache /root/.cache
EXPOSE 8000
CMD ["personal-assistant", "server"]⏎
