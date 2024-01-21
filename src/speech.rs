use std::{collections::HashMap, fs::read_to_string};

use anyhow::{anyhow, Result};
use candle_core::{cpu_backend::CpuDevice, Device::Cpu, Tensor};
use candle_onnx::read_file;
use hf_hub::{api::sync::Api, Repo, RepoType};

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
    println!("parsing lexicon");
    let lexicon = build_map(read_to_string(repo.get("lexicon.txt")?)?);
    println!("parsing tokens");
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
    println!("{:?}", punct);

    let input = "Hello, my name is Persephone!".to_lowercase();
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
    let device = Cpu;
    let tensor = Tensor::from_vec(acc.clone(), acc.len(), &device)?;
    println!("{:#?}", graph.input);

    let x_length = Tensor::from_vec(vec![acc.len() as i64], 1, &device)?;
    let noise_scale = Tensor::from_vec(vec![1 as f32], 1, &device)?;
    let noise_scale_w = Tensor::from_vec(vec![1 as f32], 1, &device)?;
    let length_scale = Tensor::from_vec(vec![1 as f32], 1, &device)?;
    let sid = Tensor::from_vec(vec![0 as i64], 1, &device)?;

    let mut inputs: HashMap<String, Tensor> = std::collections::HashMap::new();
    println!("{:#?}", graph.input);
    inputs.insert("x".to_string(), tensor.unsqueeze(0)?);
    inputs.insert("x_length".to_string(), x_length);
    inputs.insert("sid".to_string(), sid);
    inputs.insert("noise_scale".to_string(), noise_scale);
    inputs.insert("noise_scale_w".to_string(), noise_scale_w);
    inputs.insert("length_scale".to_string(), length_scale);
    let outputs = candle_onnx::simple_eval(&model, inputs)?;
    println!("{:?}", outputs);
    Ok(())
}
