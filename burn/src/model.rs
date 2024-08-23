use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
	nn::{
		conv::{
			Conv2d,
			Conv2dConfig
		},
		pool::{
			AdaptiveAvgPool2d,
			AdaptiveAvgPool2dConfig
		},
		Dropout, DropoutConfig, 
		Linear, LinearConfig, 
		Relu
	},
	prelude::*
};
use nn::loss::CrossEntropyLoss;

use crate::dataset::MnistBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
	conv1: Conv2d<B>, 
	conv2: Conv2d<B>,
	pool: AdaptiveAvgPool2d, 
	dropout: Dropout, 
	linear1: Linear<B>, 
	linear2: Linear<B>,
	activation: Relu
}

#[derive(Debug, Config)]
pub struct ModelConfig {
	num_classes: usize, 
	hidden_size: usize, 
	#[config(default = "0.5")]
	dropout: f64
}

impl ModelConfig {
	pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
		Model { 
			conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device), 
			conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
			pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(), 
			dropout: DropoutConfig::new(self.dropout).init(), 
			linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device), 
			linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device), 
			activation: Relu::new() 
		}
	}
}

impl<B: Backend> Model<B> {
	pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
		let [batch_size, height, width] = images.dims();
		
		let x = images.reshape([batch_size, 1, height, width]);
		
		let x = self.conv1.forward(x);
		let x = self.dropout.forward(x);
		let x = self.conv2.forward(x);
		let x = self.dropout.forward(x);
		let x = self.activation.forward(x);
		
		let x = self.pool.forward(x);
		let x = x.reshape([batch_size, 16 * 8 * 8]);
		let x = self.linear1.forward(x);
		let x = self.dropout.forward(x);
		let x = self.activation.forward(x);
		
		return self.linear2.forward(x);
	}
	
	pub fn forward_classification(
		&self,
		images: Tensor<B, 3>,
		targets: Tensor<B, 1, Int>
	) -> ClassificationOutput<B> {
		let output = self.forward(images);
		let loss = CrossEntropyLoss::new(
			None, 
			&output.device()
		).forward(output.clone(), targets.clone());
		
		return ClassificationOutput::new(loss, output, targets);
	}
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}