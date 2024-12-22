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
    let mut assistant = Assistant::new(model, tokenizer, config);
    let prompt = SimplePrompt::new();

    let result = prompt
        .run(
            &mut assistant,
            Some(String::from(
                "<|im_start|>system
Reply to all questions with your name 'Persephone'<|im_end|>
<|im_start|>user
What is your name?<|im_end|>assistant
",
            )),
        )
        .await
        .unwrap();

    assert_eq!(result, String::from("My name is Persephone."));
}
