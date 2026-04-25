use std::io::Result;

fn main() -> Result<()> {
    // Compile the onnx.proto file into Rust code.
    // The output is stored in the generic OUT_DIR env variable.
    prost_build::compile_protos(&["src/proto/onnx.proto"], &["src/proto/"])?;
    Ok(())
}
