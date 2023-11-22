use personal_assistant::assistant::Assistant;
use personal_assistant::loading::{ModelFile, TokenizerFile};

#[test]
#[ignore]
fn assistant_works() {
    let tokenizer = TokenizerFile::download().unwrap().tokenizer().unwrap();
    let model = ModelFile::download().unwrap().model().unwrap();
    let mut assistant = Assistant::new(model, tokenizer);
    let result = assistant
        .answer(
            "Your name is Assistant. You only answer with one word, your name.\nUSER: Say your name?\nASSISTANT:",
        )
        .unwrap();
    assert_eq!(
        result,
        String::from(
            "Your name is Assistant. You only answer with one word, your name.\nUSER: Say your name?\nASSISTANT: Assistant."
        )
    );
}
