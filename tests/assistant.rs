use personal_assistant::assistant::Assistant;
use personal_assistant::loading::{ModelFile, TokenizerFile};

// This test is really expensive
#[test]
fn assistant_works() {
    let tokenizer = TokenizerFile::download().unwrap().tokenizer().unwrap();
    let model = ModelFile::download().unwrap().model().unwrap();
    let mut assistant = Assistant::new(model, tokenizer);
    let result = assistant
        .answer(
            r#"<|system|>
Your name is Assistant. You only know one word: Assistant. Reply with that single word.</s>
<|user|>
What is your name?</s>
<|assistant|>
"#,
        )
        .unwrap();
    assert_eq!(
        result,
        String::from(
            "Your name is Assistant. You only know one word: your name Assistant. Answer every question only with the word 'Assistant'.\nUSER: Say your name?\nASSISTANT: Assistant."
        )
    );
}
