//! FN3-30 (#76) diagnostic turned regression probe: the trained GPT-2 on the
//! resident Vulkan default.
//!
//! Before FN3-26 (#72)'s identity-`Reshape` fold, this run failed inside
//! `Session::prefill` with `NotImplemented { what: "reading a device-resident
//! tensor through the host slice API" }`: the trained `Gather` consumed
//! `Reshape(input_ids, [1, 13])`, and the planner had placed that reshape's
//! output in a device-resident slot. The wrapper delegates every `Backend` call
//! to the real Vulkan backend and prints the op and operand metadata, so a
//! future regression still names the instruction that failed.
//!
//! The probe now asserts the prefill succeeds and its top-1 is the pinned
//! `640`. It does not set `FN3_VK_STAGED`; the default (resident) path is the
//! subject.
//!
//! ```bash
//! FN3_TRAINED_ONNX=/path/to/model.onnx FN3_VK_DEVICE_UUID=... \
//!   cargo test --release --features vulkan --test fn3_vulkan_trained_probe \
//!   -- --ignored --nocapture
//! ```

#![cfg(feature = "vulkan")]

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::fn3::device::DeviceRequest;
use fastnn::fn3::import::import_onnx_file;
use fastnn::fn3::model::{ModelWeights, Planner, PreparedModel, Session};
use fastnn::fn3::shape::Shape;
use fastnn::fn3::{
    Backend, Capabilities, DType, DeviceId, Fn3Error, LiveDims, OpId, OpParams, Support, Tensor,
    TensorMeta, TensorMut, TensorView,
};

const PROMPT: [u32; 13] = [
    464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257,
];

#[derive(Clone)]
struct LoggingBackend {
    inner: Arc<VulkanBackend>,
}

impl fmt::Debug for LoggingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoggingBackend").finish_non_exhaustive()
    }
}

impl Backend for LoggingBackend {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn device(&self) -> DeviceId {
        self.inner.device()
    }

    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    fn supports(&self, op: OpId) -> Support {
        self.inner.supports(op)
    }

    fn admits(&self, op: OpId, shapes: &[Shape], dtypes: &[DType], params: &OpParams) -> Support {
        self.inner.admits(op, shapes, dtypes, params)
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        let describe = |index: usize, view: &TensorView<'_>| {
            format!(
                "in{index}={} {} resident={} contiguous={} offset={}",
                view.shape(),
                view.dtype(),
                view.is_device_resident(),
                view.meta().is_contiguous(),
                view.meta().offset()
            )
        };
        let mut summary: Vec<String> = inputs
            .iter()
            .enumerate()
            .map(|(index, view)| describe(index, view))
            .collect();
        for (index, output) in outputs.iter().enumerate() {
            summary.push(format!(
                "out{index}={} {} resident={} contiguous={}",
                output.shape(),
                output.dtype(),
                output.is_device_resident(),
                output.is_contiguous()
            ));
        }
        match self.inner.execute_op(op, inputs, outputs, params) {
            Ok(()) => Ok(()),
            Err(error) => {
                println!("FAILED op {op} params {params:?}: {summary:?}");
                println!("  error: {error}");
                Err(error)
            }
        }
    }

    fn supports_resident_storage(&self) -> bool {
        self.inner.supports_resident_storage()
    }

    fn allocate_resident(&self, shape: &Shape, dtype: DType) -> Result<Tensor, Fn3Error> {
        self.inner.allocate_resident(shape, dtype)
    }

    fn prepare_persistent(&self, tensors: &[&Tensor]) -> Result<(), Fn3Error> {
        self.inner.prepare_persistent(tensors)
    }

    fn begin_step(&self, label: &'static str, batch: bool) -> Result<(), Fn3Error> {
        self.inner.begin_step(label, batch)
    }

    fn write_resident(
        &self,
        dst: &fastnn::fn3::Storage,
        byte_offset: u64,
        bytes: &[u8],
    ) -> Result<(), Fn3Error> {
        self.inner.write_resident(dst, byte_offset, bytes)
    }

    fn copy_resident_regions(
        &self,
        dst: &fastnn::fn3::Storage,
        src: &fastnn::fn3::Storage,
        regions: &[(u64, u64, u64)],
    ) -> Result<(), Fn3Error> {
        self.inner.copy_resident_regions(dst, src, regions)
    }

    fn note_readback(&self, bytes: u64) {
        self.inner.note_readback(bytes)
    }

    fn end_step(&self) -> Result<u32, Fn3Error> {
        self.inner.end_step()
    }
}

fn retag_weights(weights: &ModelWeights, device: DeviceId) -> ModelWeights {
    let mut named = Vec::with_capacity(weights.len());
    for index in 0..weights.len() {
        let tensor = weights.get(index).expect("weight");
        let meta = TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)
            .expect("weight metadata");
        named.push((
            weights.name(index).expect("weight name").to_string(),
            Tensor::from_storage(tensor.storage().clone(), meta).expect("retag"),
        ));
    }
    ModelWeights::with_names(named).expect("weights")
}

fn live_dims(seq: u64, past: u64) -> LiveDims {
    LiveDims::new(1, seq, past, 12, 12, 64).expect("dims")
}

#[test]
#[ignore = "needs the trained ONNX graph and a physical Vulkan device"]
fn resident_vulkan_trained_prefill_names_the_failing_op() {
    let Some(path) = std::env::var("FN3_TRAINED_ONNX").ok().map(PathBuf::from) else {
        println!("SKIP: FN3_TRAINED_ONNX is not set");
        return;
    };
    let Some(index) = adapters()
        .iter()
        .find(|adapter| {
            adapter.has_compute_queue()
                && !adapter.is_software()
                && std::env::var("FN3_VK_DEVICE_UUID")
                    .ok()
                    .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid.trim()))
                    .unwrap_or(true)
        })
        .map(|adapter| adapter.index)
    else {
        println!("SKIP: no physical Vulkan compute device");
        return;
    };
    let inner = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    )
    .expect("open device");
    let device = inner.device_id();
    let backend = LoggingBackend { inner };
    let import = import_onnx_file(&path).expect("import");
    let cpu = fastnn::fn3::CpuBackend::new();
    let planner = Planner::new(&import, &cpu);
    let envelope = planner.envelope(1, 13, 32, 33).expect("envelope");
    let config = import.geometry().kv_cache_config(32, 1);
    let weights = retag_weights(&import.to_model_weights().expect("weights"), device);
    let plan = planner
        .plan(&envelope, live_dims(13, 0))
        .expect("plan")
        .execution()
        .clone();
    let model = PreparedModel::new(plan, weights, device, envelope, Some(config)).expect("model");
    let mut session = Session::with_runtime(&model, &import, &backend).expect("session");
    match session.prefill(&PROMPT) {
        Ok(logits) => {
            let host = logits.to_host().expect("download");
            let values = host.as_f32_slice().expect("logits");
            let argmax = values
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 {
                        (i, v)
                    } else {
                        best
                    }
                })
                .0;
            println!("resident Vulkan trained prefill succeeded, argmax {argmax}");
            assert_eq!(
                argmax, 640,
                "the pinned trained prompt's argmax changed on the resident path"
            );
        }
        Err(error) => {
            println!("resident Vulkan trained prefill failed: {error}");
            panic!("resident Vulkan trained prefill failed: {error}");
        }
    }
}
