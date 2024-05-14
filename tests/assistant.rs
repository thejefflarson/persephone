use persephone::assistant::Assistant;
use persephone::loading::{ModelFile, TokenizerFile};
use persephone::prompt::BlockingPrompt;
use persephone::prompt::SimplePrompt;

// This test is really expensive
#[ignore]
#[tokio::test]
async fn assistant_works() {
    let tokenizer = TokenizerFile::download().unwrap().tokenizer().unwrap();
    let model = ModelFile::download().unwrap().model().unwrap();
    let mut assistant = Assistant::new(model, tokenizer);
    let prompt = SimplePrompt::new();

    let result = prompt
        .run(
            &mut assistant,
            Some(String::from(
                "<|system|>Reply to all questions with 'Persephone'.</s><|user|>What is your name?</s><|assistant|>",
            )),
        )
        .await
        .unwrap();

    assert_eq!(result, String::from("Persephone"));
}
