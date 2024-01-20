use personal_assistant::assistant::Assistant;
use personal_assistant::loading::{ModelFile, TokenizerFile};
use personal_assistant::prompt::BlockingPrompt;
use personal_assistant::prompt::SimplePrompt;

// This test is really expensive
#[tokio::test]
async fn assistant_works() {
    let tokenizer = TokenizerFile::download().unwrap().tokenizer().unwrap();
    let model = ModelFile::download().unwrap().model().unwrap();
    let assistant = Assistant::new(model, tokenizer);
    let prompt = SimplePrompt::new();

    let result = prompt
        .run(
            &assistant,
            Some(String::from(
                "Input: Say only the word Assistant. Do not reply with any other text.\nOutput:",
            )),
        )
        .await
        .unwrap();

    assert_eq!(
        result,
        String::from("Input: Say only the word Assistant. Do not reply with any other text.\nOutput: Assistant.\n")
    );
}
