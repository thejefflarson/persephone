use std::{collections::HashMap, fs::read_to_string, io::Write};

use anyhow::{anyhow, Result};
use candle_core::{cpu_backend::CpuDevice, Device::Cpu};
use candle_onnx::read_file;
use hf_hub::{api::sync::Api, Repo, RepoType};
use hound::{Sample, WavSpec};
use ndarray::{Array1, Axis};
use ort::{inputs, Session, Tensor};

use crate::utils;

// from https://huggingface.co/csukuangfj/vits-ljs/blob/main/test.py
fn build_map(file: String) -> HashMap<String, String> {
    let mut ret: HashMap<String, String> = HashMap::new();
    for line in file.split("\n").collect::<Vec<&str>>() {
        let bits = line.trim_start().split(" ").collect::<Vec<&str>>();
        let k = bits[0].to_string();
        if bits.len() == 1 && bits[0] != "" {
            ret.insert(" ".to_string(), k.to_string());
            continue;
        }
        let v = bits[1..].join(" ").to_string();

        ret.insert(k, v);
    }
    ret
}
// using ort until candle implements Gather: https://github.com/huggingface/candle/issues/1305
pub fn say() -> Result<()> {
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        "csukuangfj/vits-ljs".into(),
        RepoType::Model,
        "main".into(),
    ));
    //println!("parsing lexicon");
    let lexicon = build_map(read_to_string(repo.get("lexicon.txt")?)?);
    //println!("parsing tokens");
    let tok_map = build_map(read_to_string(repo.get("tokens.txt")?)?);

    let model = read_file(repo.get("vits-ljs.onnx")?)?;
    let graph = model.graph.as_ref().unwrap();
    let punctuation = model
        .metadata_props
        .iter()
        .filter(|it| it.key == "punctuation")
        .map(|it| {
            it.value
                .to_string()
                .split(" ")
                .map(|e| e.to_string())
                .collect::<Vec<String>>()
        })
        .collect::<Vec<Vec<String>>>();
    let punct = &punctuation[0];
    //println!("{:?}", punct);

    let input = "Hello, my name is Persephone! Most common audio formats can be streamed using specific server-side technologies.

Note: It's potentially easier to stream audio using non-streaming formats because unlike video there are no keyframes.".to_lowercase();
    let space = tok_map
        .get(&" ".to_string())
        .ok_or(anyhow!("no space character?"))?
        .parse::<i64>()?;
    let mut acc: Vec<i64> = vec![];
    for word in input.split(" ") {
        if let Some(phonemes) = lexicon.get(word) {
            let mut codes = phonemes
                .split(" ")
                .flat_map(|phone| tok_map.get(phone))
                .map(|code| code.parse::<i64>())
                .flatten()
                .collect::<Vec<i64>>();
            acc.append(&mut codes);
            acc.push(space);
        }
    }

    let model = Session::builder()?.with_model_from_file(repo.get("vits-ljs.onnx")?)?;
    //println!("{:#?}", model.inputs);
    //println!("{:#?}", model.metadata()?.custom("sample_rate"));
    //println!("{:?}", acc);
    let ndarray = Array1::from_iter(acc.iter().cloned()).insert_axis(Axis(0));
    let x_length = Array1::from_vec(vec![1 as i64]);
    let noise_scale = Array1::from_vec(vec![1 as f32]);
    let noise_scale_w = Array1::from_vec(vec![1 as f32]);
    let length_scale = Array1::from_vec(vec![1 as f32]);

    let outputs = model.run(inputs![
        "x" => ndarray,
        "x_length" => x_length,
        "noise_scale" => noise_scale,
        "noise_scale_w" => noise_scale_w,
        "length_scale" => length_scale,
    ]?)?;

    //println!("{:?}", outputs["y"]);
    let outputs: Tensor<f32> = outputs["y"].extract_tensor()?;
    let outputs = outputs.view();
    //println!("{:?}", outputs.clone().into_iter().collect::<Vec<&f32>>());

    // https://github.com/ruuda/hound/blob/master/examples/append.rs
    let spec = WavSpec {
        bits_per_sample: 32,
        channels: 1,
        sample_format: hound::SampleFormat::Float,
        sample_rate: 22050,
    };

    let v = spec.into_header_for_infinite_file();

    let so = std::io::stdout();
    let mut so = so.lock();
    so.write_all(&v[..]).unwrap();

    for sample in outputs.iter() {
        Sample::write(sample.to_owned(), &mut so, 32).unwrap();
    }
    Ok(())
}
