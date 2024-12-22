FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
RUN wget https://github.com/mozilla/sccache/releases/download/v0.9.0/sccache-v0.9.0-armv7-unknown-linux-musleabi.tar.gz \
    && tar xzvf sccache-v0.9.0-armv7-unknown-linux-musleabi.tar.gz \
    && mv sccache-v0.9.0-armv7-unknown-linux-musleabi/sccache /usr/local/bin/sccache \
    && chmod +x /usr/local/bin/sccache
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache
WORKDIR /app

FROM chef AS planner
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json
# Build application
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --release
RUN /app/target/release/persephone download

FROM rust:1
COPY --from=builder /app/target/release/persephone /usr/local/bin/persephone
COPY --from=builder /root/.cache /root/.cache
EXPOSE 8000
CMD ["persephone", "server"]⏎
