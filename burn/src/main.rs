mod model;
mod dataset;
mod training;

use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistItem;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Record, Recorder};
use burn::tensor::ElementConversion;
use burn::train::metric::Adaptor;
use dataset::MnistBatcher;
use model::ModelConfig;
use training::TrainingConfig;

pub fn infer<B: Backend>(
	artifact_dir: &str, 
	device: B::Device, 
	item: MnistItem
) -> bool {
	
	let config = TrainingConfig::load(
		format!("{artifact_dir}/config.json")
	)
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output
    	.argmax(1)
     	.flatten::<1>(0, 1)
      	.into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
    
    if label == predicted.elem::<u8>() {
    	return true;
    } else {
    	return false;
    }
}

fn main() {
	type Backend = Wgpu<AutoGraphicsApi, f32, i32>;
	type AutodiffBackend = Autodiff<Backend>;
	
	let device = burn::backend::wgpu::WgpuDevice::default();
	
	// // uncommment this for training a model 
	// crate::training::train::<AutodiffBackend>(
	// 	"/tmp/guide", 
	// 	crate::training::TrainingConfig::new(
	// 		ModelConfig::new(10, 512), 
	// 		AdamConfig::new()
	// 	), 
	// 	device.clone()
	// );
	
	let test_set = burn::data::dataset::vision::MnistDataset::test();
	let mut failed_tests: usize = 0;
	
	for index in 0..=100 {
		let test_result = infer::<Backend>(
	        "/tmp/guide",
	        device.clone(),
			test_set
				.get(index)
				.unwrap()
	    );
		
		if test_result == false {
			failed_tests += 1;
		} else {
			continue;
		}
	}
	
	println!("{}", failed_tests);
	
}