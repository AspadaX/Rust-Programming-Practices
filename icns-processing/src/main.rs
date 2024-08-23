use std::any::Any;

use icns;

fn main() {
	let target_path = "/Users/xinyubao/Downloads/macos-1/".to_string();
	
	let file = std::fs::File::open("/Users/xinyubao/Downloads/macos-1/AppIcon.icns")
		.unwrap();
	let buffer = std::io::BufReader::new(file);
	
	let mut icons = icns::IconFamily::read(buffer)
		.unwrap();
	
	let available_icons = &icons.available_icons();
	
	for icon in available_icons {
		let image = &icons.get_icon_with_type(icon.clone())
			.unwrap();
		
		let mut target_filepath = target_path.clone();
			
		target_filepath.push_str(
			format!("{}.png", icon.ostype().to_string())
				.as_str()
		);
		
		let target_file = std::fs::File::create_new(&target_filepath)
			.unwrap();
		
		image.write_png(target_file).unwrap();
	}
}
