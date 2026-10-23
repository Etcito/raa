//! Adapter, device and queue: [`GpuContext`], [`AdapterSummary`], [`GpuOptions`] and
//! [`DeviceRequest`].

use crate::RaaError;

/// Substrings (lower case) that mark a software adapter (the Conventions hardware rule).
const SOFTWARE_MARKERS: [&str; 5] = [
    "swiftshader",
    "llvmpipe",
    "lavapipe",
    "software",
    "basic render",
];

/// What the adapter says about itself, in the BrowserResult vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterSummary {
    pub vendor: String,
    pub architecture: String,
    pub device: String,
    pub description: String,
    pub backend: String,
    pub is_fallback: bool,
}

impl AdapterSummary {
    /// Builds the summary from `adapter.get_info()`.
    pub fn from_info(info: &wgpu::AdapterInfo) -> Self {
        let description = match (info.driver.is_empty(), info.driver_info.is_empty()) {
            (false, false) => format!("{} {}", info.driver, info.driver_info),
            (false, true) => info.driver.clone(),
            (true, false) => info.driver_info.clone(),
            (true, true) => String::new(),
        };
        // wgpu exposes no vendor name, only a PCI vendor id (0 when unknown, e.g. on the web).
        let vendor = if info.vendor == 0 {
            String::new()
        } else {
            format!("0x{:04x}", info.vendor)
        };
        AdapterSummary {
            vendor,
            architecture: String::new(),
            device: info.name.clone(),
            description,
            backend: format!("{:?}", info.backend),
            // The WebGPU backend maps `isFallbackAdapter` to `DeviceType::Cpu`.
            is_fallback: info.device_type == wgpu::DeviceType::Cpu,
        }
    }

    /// `!is_fallback` and no software marker in vendor, architecture, device or description
    /// (case-insensitive substring match).
    pub fn is_hardware(&self) -> bool {
        if self.is_fallback {
            return false;
        }
        [
            &self.vendor,
            &self.architecture,
            &self.device,
            &self.description,
        ]
        .iter()
        .all(|field| {
            let lower = field.to_lowercase();
            !SOFTWARE_MARKERS.iter().any(|m| lower.contains(m))
        })
    }
}

/// How to pick the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOptions {
    pub power_preference: wgpu::PowerPreference,
    pub allow_software: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            allow_software: true,
        }
    }
}

/// What to ask of the device beyond the core WebGPU defaults.
///
/// `optional_features` are enabled only where the adapter has them (never required);
/// `adapter_limits` requests the adapter's own limits instead of the default limits.
#[derive(Clone, Debug, Default)]
pub struct DeviceRequest {
    pub optional_features: wgpu::Features,
    pub adapter_limits: bool,
}

/// The adapter, device and queue used by every other RAA object.
#[derive(Debug)]
pub struct GpuContext {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    summary: AdapterSummary,
}

impl GpuContext {
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn summary(&self) -> &AdapterSummary {
        &self.summary
    }

    /// The device's enabled features.
    pub fn features(&self) -> wgpu::Features {
        self.device.features()
    }

    /// The device's limits.
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    async fn from_adapter(
        adapter: wgpu::Adapter,
        req: &DeviceRequest,
    ) -> Result<GpuContext, RaaError> {
        let summary = AdapterSummary::from_info(&adapter.get_info());
        let required_limits = if req.adapter_limits {
            adapter.limits()
        } else {
            wgpu::Limits::default()
        };
        let desc = wgpu::DeviceDescriptor {
            label: Some("raa-device"),
            required_features: req.optional_features & adapter.features(),
            required_limits,
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(&desc)
            .await
            .map_err(|e| RaaError::RequestDevice(e.to_string()))?;
        Ok(GpuContext {
            adapter,
            device,
            queue,
            summary,
        })
    }

    /// WebGPU context and surface for a canvas (WebGPU backend only), default device request.
    #[cfg(target_arch = "wasm32")]
    pub async fn for_canvas(
        canvas: web_sys::HtmlCanvasElement,
        opts: &GpuOptions,
    ) -> Result<(GpuContext, crate::CanvasSurface), RaaError> {
        GpuContext::for_canvas_with(canvas, opts, &DeviceRequest::default()).await
    }

    /// WebGPU context and surface for a canvas: a core adapter (never the compatibility
    /// feature level), then a device with `req.optional_features & adapter.features()` and,
    /// when `req.adapter_limits`, the adapter's limits.
    #[cfg(target_arch = "wasm32")]
    pub async fn for_canvas_with(
        canvas: web_sys::HtmlCanvasElement,
        opts: &GpuOptions,
        req: &DeviceRequest,
    ) -> Result<(GpuContext, crate::CanvasSurface), RaaError> {
        if !navigator_gpu_present() {
            return Err(RaaError::NoAdapter(
                "navigator.gpu is unavailable (WebGPU disabled or unsupported)".to_string(),
            ));
        }
        let (width, height) = (canvas.width(), canvas.height());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| RaaError::Surface(e.to_string()))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: opts.power_preference,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| RaaError::NoAdapter(e.to_string()))?;
        let gpu = GpuContext::from_adapter(adapter, req).await?;
        if !opts.allow_software && !gpu.summary.is_hardware() {
            return Err(RaaError::NoAdapter(format!(
                "only a software adapter is available ({}) and allow_software is false",
                gpu.summary.device
            )));
        }
        let surface = crate::CanvasSurface::new(&gpu, surface, width, height)?;
        Ok((gpu, surface))
    }

    /// Headless context on any native backend. When no adapter is found and
    /// `allow_software` is set, retries with `force_fallback_adapter = true`.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn headless(opts: &GpuOptions) -> Result<GpuContext, RaaError> {
        GpuContext::headless_with(opts, &DeviceRequest::default()).await
    }

    /// Headless context like [`GpuContext::headless`], with the device built from `req`
    /// (same rules as `for_canvas_with`).
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn headless_with(
        opts: &GpuOptions,
        req: &DeviceRequest,
    ) -> Result<GpuContext, RaaError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let request = |force_fallback_adapter| wgpu::RequestAdapterOptions {
            power_preference: opts.power_preference,
            force_fallback_adapter,
            compatible_surface: None,
            ..Default::default()
        };
        let adapter = match instance.request_adapter(&request(false)).await {
            Ok(adapter) => adapter,
            Err(first) if opts.allow_software => instance
                .request_adapter(&request(true))
                .await
                .map_err(|e| {
                    RaaError::NoAdapter(format!("{first}; with force_fallback_adapter: {e}"))
                })?,
            Err(e) => return Err(RaaError::NoAdapter(e.to_string())),
        };
        let gpu = GpuContext::from_adapter(adapter, req).await?;
        if !opts.allow_software && !gpu.summary.is_hardware() {
            return Err(RaaError::NoAdapter(format!(
                "only a software adapter is available ({}) and allow_software is false",
                gpu.summary.device
            )));
        }
        Ok(gpu)
    }
}

/// True when `navigator.gpu` exists and is neither undefined nor null.
#[cfg(target_arch = "wasm32")]
fn navigator_gpu_present() -> bool {
    let global = js_sys::global();
    let Ok(navigator) = js_sys::Reflect::get(&global, &"navigator".into()) else {
        return false;
    };
    if navigator.is_undefined() || navigator.is_null() {
        return false;
    }
    match js_sys::Reflect::get(&navigator, &"gpu".into()) {
        Ok(gpu) => !gpu.is_undefined() && !gpu.is_null(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary(vendor: &str, device: &str, description: &str, fallback: bool) -> AdapterSummary {
        AdapterSummary {
            vendor: vendor.to_string(),
            architecture: String::new(),
            device: device.to_string(),
            description: description.to_string(),
            backend: "Vulkan".to_string(),
            is_fallback: fallback,
        }
    }

    #[test]
    fn swiftshader_is_software() {
        let s = summary("google", "SwiftShader Device (Subzero)", "", false);
        assert!(!s.is_hardware());
    }

    #[test]
    fn llvmpipe_is_software() {
        let s = summary(
            "0x10005",
            "llvmpipe (LLVM 17.0.6, 256 bits)",
            "Mesa 24.0",
            false,
        );
        assert!(!s.is_hardware());
    }

    #[test]
    fn microsoft_basic_render_driver_description_is_software() {
        let s = summary("0x1414", "Adapter", "Microsoft Basic Render Driver", false);
        assert!(!s.is_hardware());
    }

    #[test]
    fn fallback_adapter_is_software() {
        let s = summary("nvidia", "NVIDIA GeForce RTX 3090", "", true);
        assert!(!s.is_hardware());
    }

    #[test]
    fn nvidia_adapter_is_hardware() {
        let s = summary("0x10de", "NVIDIA GeForce RTX 3090", "NVIDIA 560.94", false);
        assert!(s.is_hardware());
    }

    #[test]
    fn default_device_request_asks_for_nothing_extra() {
        let r = DeviceRequest::default();
        assert_eq!(r.optional_features, wgpu::Features::empty());
        assert!(!r.adapter_limits);
    }

    #[test]
    fn default_options_prefer_high_performance_and_allow_software() {
        let o = GpuOptions::default();
        assert_eq!(o.power_preference, wgpu::PowerPreference::HighPerformance);
        assert!(o.allow_software);
    }
}
