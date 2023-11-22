use anyhow::Result;
use clap::{Parser, ValueEnum};
use personal_assistant::loading::{ModelFile, TokenizerFile};
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Command {
    /// Download and cache models
    Download,
    /// Serve the api
    Serve,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Which mode to run in
    #[arg(value_enum)]
    command: Command,
}

fn download() -> Result<()> {
    let filename = ModelFile::download()?;
    let tokenizer = TokenizerFile::download()?;
    println!("Model saved in {} and tokenizer in {}", filename, tokenizer);
    Ok(())
}

fn serve() -> Result<()> {
    let model = ModelFile::download()?.model()?;
    let tokenizer = TokenizerFile::download()?.tokenizer()?;
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Download => {
            download().expect("couldn't get models");
        }
        Command::Serve => {
            serve().expect("couldn't start server");
        }
    }
}
