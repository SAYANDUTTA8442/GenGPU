//! Physical NVIDIA PTX module lifecycle smoke test.
//!
//! Run with:
//! cargo run -p gengpu-driver --example nvidia_module_smoke

use gengpu_core::ErrorKind;
use gengpu_driver::{nvidia::NvidiaDriver, Driver};

const PTX: &str = r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry gengpu_module_smoke()
{
    ret;
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let driver = NvidiaDriver::new();
    driver.initialize()?;

    let devices = driver.enumerate_devices()?;
    let device = devices
        .first()
        .ok_or("no NVIDIA CUDA device is available")?;
    println!("CUDA driver version: {}", driver.driver_version()?);
    println!("GPU: {}", device.name);

    let mut context = driver.create_context(device.id)?;
    let mut module = context.load_ptx(PTX)?;

    let function = module.get_function("gengpu_module_smoke")?;
    drop(function);
    println!("PTX module load and function lookup: PASS");

    let close_error = context
        .close()
        .expect_err("context close must reject a live PTX module");
    if close_error.kind() != ErrorKind::InvalidState {
        return Err(format!("unexpected live-module close error: {close_error}").into());
    }
    println!("live-module context-close restriction: PASS");

    module.unload()?;
    println!("explicit module unload: PASS");

    let dropped_module = context.load_ptx(PTX)?;
    drop(dropped_module);
    println!("best-effort module Drop: completed");

    context.close()?;
    println!("context close after module cleanup: PASS");
    Ok(())
}
