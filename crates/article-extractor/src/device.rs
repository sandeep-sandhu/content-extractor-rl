//! Device selection for CPU/CUDA

use candle_core::Device;
use tracing::{info, warn};

/// Get best available device (CUDA if available, otherwise CPU)
pub fn get_device() -> Device {
    // Check environment variable for forcing CPU
    if std::env::var("ARTICLE_EXTRACTOR_FORCE_CPU").is_ok() {
        info!("ARTICLE_EXTRACTOR_FORCE_CPU set, using CPU");
        return Device::Cpu;
    }

    // Try to use CUDA if available
    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            match Device::new_cuda(0) {
                Ok(device) => {
                    println!("Using CUDA device (GPU)");
                    println!("Training will use GPU acceleration");
                    return device;
                }
                Err(e) => {
                    warn!("CUDA available but failed to initialize: {}. Falling back to CPU", e);
                }
            }
        } else {
            info!("CUDA not available, using CPU");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        info!("Using CPU (built without CUDA support)");
    }

    Device::Cpu
}

/// Get device with preference (for testing/debugging)
pub fn get_device_with_preference(prefer_cpu: bool) -> Device {
    if prefer_cpu {
        info!("Using CPU (forced)");
        return Device::Cpu;
    }

    get_device()
}

/// Check if CUDA is available
pub fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        candle_core::utils::cuda_is_available()
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get device info string
pub fn get_device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => {
            // CudaDevice in candle doesn't expose device ID directly
            // We just indicate it's using CUDA
            "CUDA GPU".to_string()
        }
        Device::Metal(_) => "Metal GPU".to_string(),
    }
}

/// Print device information
pub fn print_device_info() {
    let device = get_device();
    let info = get_device_info(&device);

    println!("╔════════════════════════════════════════╗");
    println!("║   Article Extractor - Device Info      ║");
    println!("╠════════════════════════════════════════╣");

    #[cfg(feature = "cuda")]
    println!("║ Build: CUDA support enabled            ║");
    #[cfg(not(feature = "cuda"))]
    println!("║ Build: CPU only (no CUDA)              ║");

    println!("║ Runtime: {:<30}║", info);

    if cuda_is_available() {
        println!("║ Status:   GPU acceleration active      ║");
    } else {
        println!("║ Status:   CPU mode                    ║");
    }

    println!("╚════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection() {
        let device = get_device();
        println!("Selected device: {:?}", device);

        let info = get_device_info(&device);
        println!("Device info: {}", info);

        println!("CUDA available: {}", cuda_is_available());
    }

    #[test]
    fn test_force_cpu() {
        std::env::set_var("ARTICLE_EXTRACTOR_FORCE_CPU", "1");
        let device = get_device();
        assert!(matches!(device, Device::Cpu));
        std::env::remove_var("ARTICLE_EXTRACTOR_FORCE_CPU");
    }

    #[test]
    fn test_device_info_cpu() {
        let device = Device::Cpu;
        let info = get_device_info(&device);
        assert_eq!(info, "CPU");
    }
}