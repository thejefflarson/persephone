use personal_assistant::assistant::Assistant;
use personal_assistant::loading::{ModelFile, TokenizerFile};

// This test is really expensive
#[test]
fn assistant_works() {
    let tokenizer = TokenizerFile::download().unwrap().tokenizer().unwrap();
    let model = ModelFile::download().unwrap().model().unwrap();
    let mut assistant = Assistant::new(model, tokenizer);
    let result = assistant
        .answer("Input: Say only the word Assistant. Do not reply with any other text.\nOutput:")
        .unwrap();
    assert_eq!(
        result,
        String::from("Input: Say only the word Assistant. Do not reply with any other text.\nOutput: Assistant.\n")
    );
}
