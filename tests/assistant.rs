use persephone::assistant::Assistant;
use persephone::loading::{ModelFile, TokenizerFile};
use persephone::prompt::BlockingPrompt;
use persephone::prompt::SimplePrompt;

// This test is really expensive
#[ignore]
#[tokio::test]
async fn assistant_works() {
    let tokenizer = TokenizerFile::download().unwrap().tokenizer().unwrap();
    let (model, config) = ModelFile::download().unwrap().model().unwrap();
    let mut assistant = Assistant::new(model, config, tokenizer);
    let prompt = SimplePrompt::new();

    let result = prompt
        .run(
            &mut assistant,
            Some(String::from(
                "Reply to all questions with your name, your name is 'Persephone'. Do not include any other text other than your name 'Persephone'. What is your name?",
            )),
        )
        .await
        .unwrap();

    assert_eq!(result, String::from("Your name is \"Persephone\"."));
}
