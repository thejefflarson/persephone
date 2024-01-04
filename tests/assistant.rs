use personal_assistant::assistant::Assistant;
use personal_assistant::loading::{ConfigFile, ModelFile, TokenizerFile};

// This test is really expensive
#[test]
fn assistant_works() {
    let tokenizer = TokenizerFile::download().unwrap().tokenizer().unwrap();
    let config = ConfigFile::download().unwrap().config().unwrap();
    let model = ModelFile::download().unwrap().model(config).unwrap();
    let mut assistant = Assistant::new(model, tokenizer);
    let result = assistant
        .answer(
            "Your name is Assistant. You only know one word: your name Assistant. Answer every question only with the word 'Assistant'.\nUSER: Say your name?\nASSISTANT:",
        )
        .unwrap();
    assert_eq!(
        result,
        String::from(
            "Your name is Assistant. You only know one word: your name Assistant. Answer every question only with the word 'Assistant'.\nUSER: Say your name?\nASSISTANT: Assistant."
        )
    );
}
