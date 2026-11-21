//! Pipeline ownership and the registration-time spec check.
//!
//! [`Kernel`] is one compiled compute pipeline plus everything derived from a
//! single [`KernelSpec`]: shader module, descriptor
//! set layout, pipeline layout and pipeline. Creating it validates the
//! declaration against the committed SPIR-V (entry point, `LocalSize`,
//! descriptors, push-constant size) *and* against the device limits, so a
//! kernel registered at 1024 invocations on a device with
//! `maxComputeWorkGroupInvocations = 512` fails at registration with both
//! numbers rather than at dispatch with a validation error.
//!
//! Kernels are owned by whoever holds the `Arc<Kernel>` (the backend's cache).
//! `Drop` destroys the pipeline, layouts and module in reverse creation order.

use std::ffi::CString;
use std::fmt;
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;

use super::device::Device;
use super::error::{Error, Result};
use super::kernels::reflect::{self, ReflectedKind};
use super::kernels::{BindingKind, BindingSpec, KernelSpec};

/// A compiled compute kernel.
pub struct Kernel {
    device: Arc<Device>,
    spec: &'static KernelSpec,
    shader: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Kernel {
    /// Validate `spec`, create the shader module, layouts and pipeline.
    pub fn new(device: &Arc<Device>, spec: &'static KernelSpec) -> Result<Arc<Kernel>> {
        spec.validate()?;
        let entry = validate_against_spirv(spec)?;
        validate_against_limits(device, spec)?;

        let words = spirv_words(spec.spirv)?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: `words` is a live, 4-byte-aligned Vec<u32> containing a
        // validated SPIR-V module (magic and length checked); the create-info
        // borrows it only for this call.
        let shader =
            unsafe { device.ash().create_shader_module(&shader_info, None) }.map_err(|code| {
                Error::vk(
                    "vkCreateShaderModule",
                    code,
                    format!("{} ({} bytes of SPIR-V)", spec.name, spec.spirv.len()),
                )
            })?;
        device.ledger().create("shader_module");

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = spec
            .bindings
            .iter()
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(binding.kind.descriptor_type())
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: every binding is unique (checked by `spec.validate`) and the
        // slice is live for the call. The layout is destroyed in `Drop`.
        let set_layout = match unsafe {
            device
                .ash()
                .create_descriptor_set_layout(&set_layout_info, None)
        } {
            Ok(layout) => layout,
            Err(code) => {
                // SAFETY: nothing refers to the shader module yet.
                unsafe { device.ash().destroy_shader_module(shader, None) };
                device.ledger().destroy("shader_module");
                return Err(Error::vk(
                    "vkCreateDescriptorSetLayout",
                    code,
                    format!("{}: {} declared binding(s)", spec.name, spec.bindings.len()),
                ));
            }
        };
        device.ledger().create("descriptor_set_layout");

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(spec.push_constant_bytes);
        let push_ranges: &[vk::PushConstantRange] = if spec.push_constant_bytes > 0 {
            std::slice::from_ref(&push_range)
        } else {
            &[]
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&set_layout))
            .push_constant_ranges(push_ranges);
        // SAFETY: the set layout and push-constant range are live for the call;
        // the layout is destroyed in `Drop` after the pipeline.
        let pipeline_layout = match unsafe {
            device
                .ash()
                .create_pipeline_layout(&pipeline_layout_info, None)
        } {
            Ok(layout) => layout,
            Err(code) => {
                // SAFETY: nothing refers to these objects yet; free in
                // reverse creation order.
                unsafe {
                    device.ash().destroy_descriptor_set_layout(set_layout, None);
                    device.ash().destroy_shader_module(shader, None);
                }
                device.ledger().destroy("descriptor_set_layout");
                device.ledger().destroy("shader_module");
                return Err(Error::vk(
                    "vkCreatePipelineLayout",
                    code,
                    format!(
                        "{}: {} push-constant bytes",
                        spec.name, spec.push_constant_bytes
                    ),
                ));
            }
        };
        device.ledger().create("pipeline_layout");

        let entry_point = CString::new(entry.name.clone())
            .map_err(|_| Error::registration(spec.name, "entry point contains a NUL byte"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        // SAFETY: the shader module and pipeline layout are live; the stage
        // pointer is borrowed only for the call; the pipeline cache belongs to
        // this device. Partially created pipelines are destroyed on error.
        let pipelines = unsafe {
            device.ash().create_compute_pipelines(
                device.pipeline_cache_handle(),
                &[pipeline_info],
                None,
            )
        };
        let pipeline = match pipelines {
            Ok(pipelines) => pipelines[0],
            Err((pipelines, code)) => {
                for pipeline in pipelines {
                    if !pipeline.is_null() {
                        // SAFETY: partially created pipelines are owned by us.
                        unsafe { device.ash().destroy_pipeline(pipeline, None) };
                    }
                }
                // SAFETY: nothing refers to the layout or module yet.
                unsafe {
                    device.ash().destroy_pipeline_layout(pipeline_layout, None);
                    device.ash().destroy_descriptor_set_layout(set_layout, None);
                    device.ash().destroy_shader_module(shader, None);
                }
                device.ledger().destroy("pipeline_layout");
                device.ledger().destroy("descriptor_set_layout");
                device.ledger().destroy("shader_module");
                return Err(Error::vk(
                    "vkCreateComputePipelines",
                    code,
                    format!(
                        "{}: entry point \"{}\", {} descriptor(s), {} push-constant bytes",
                        spec.name,
                        spec.entry_point,
                        spec.bindings.len(),
                        spec.push_constant_bytes
                    ),
                ));
            }
        };
        device.ledger().create("pipeline");

        Ok(Arc::new(Kernel {
            device: Arc::clone(device),
            spec,
            shader,
            set_layout,
            pipeline_layout,
            pipeline,
        }))
    }

    /// The declaration this kernel was built from.
    pub fn spec(&self) -> &'static KernelSpec {
        self.spec
    }

    /// Stable kernel name.
    pub fn name(&self) -> &'static str {
        self.spec.name
    }

    /// Raw pipeline handle.
    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Descriptor set layout derived from the spec.
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }

    /// Pipeline layout derived from the spec.
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Declared push-constant bytes.
    pub fn push_constant_bytes(&self) -> u32 {
        self.spec.push_constant_bytes
    }

    /// Declared workgroup size.
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.spec.workgroup_size
    }

    /// Declared bindings.
    pub fn bindings(&self) -> &'static [BindingSpec] {
        self.spec.bindings
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: submitters keep the device alive and buffers/pipelines are
        // not referenced by an in-flight command buffer when their owner drops
        // them; destroyed in reverse creation order, exactly once.
        unsafe {
            self.device.ash().destroy_pipeline(self.pipeline, None);
            self.device
                .ash()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .ash()
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.ash().destroy_shader_module(self.shader, None);
        }
        self.device.ledger().destroy("pipeline");
        self.device.ledger().destroy("pipeline_layout");
        self.device.ledger().destroy("descriptor_set_layout");
        self.device.ledger().destroy("shader_module");
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernel")
            .field("name", &self.spec.name)
            .field("workgroup", &self.spec.workgroup_size)
            .field("bindings", &self.spec.bindings.len())
            .finish_non_exhaustive()
    }
}

/// Compare the declaration against the committed SPIR-V.
fn validate_against_spirv(spec: &'static KernelSpec) -> Result<reflect::ReflectedEntry> {
    let module = reflect::reflect(spec.spirv).map_err(|error| {
        Error::registration(spec.name, format!("SPIR-V reflection failed: {error}"))
    })?;
    let entry = module.entry(spec.entry_point).ok_or_else(|| {
        Error::registration(
            spec.name,
            format!(
                "entry point \"{}\" is not in the SPIR-V module; it declares [{}]",
                spec.entry_point,
                module
                    .entries
                    .iter()
                    .map(|entry| entry.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )
    })?;
    if entry.execution_model != 5 {
        return Err(Error::registration(
            spec.name,
            format!(
                "entry point \"{}\" has execution model {} (5 is GLCompute)",
                entry.name, entry.execution_model
            ),
        ));
    }
    if entry.local_size != spec.workgroup_size {
        return Err(Error::registration(
            spec.name,
            format!(
                "declared workgroup size {:?} but the SPIR-V LocalSize is {:?}",
                spec.workgroup_size, entry.local_size
            ),
        ));
    }
    if entry.push_constant_bytes != spec.push_constant_bytes {
        return Err(Error::registration(
            spec.name,
            format!(
                "declared {} push-constant bytes but the SPIR-V PushConstant block is {} bytes",
                spec.push_constant_bytes, entry.push_constant_bytes
            ),
        ));
    }

    let mut declared: Vec<(u32, u32, BindingKind)> = spec
        .bindings
        .iter()
        .map(|binding| (0, binding.binding, binding.kind))
        .collect();
    declared.sort_by_key(|(set, binding, _)| (*set, *binding));
    let reflected: Vec<(u32, u32, BindingKind)> = entry
        .bindings
        .iter()
        .map(|binding| {
            let kind = match binding.kind {
                ReflectedKind::StorageBuffer => BindingKind::StorageBuffer,
                ReflectedKind::UniformBuffer => BindingKind::UniformBuffer,
            };
            (binding.set, binding.binding, kind)
        })
        .collect();
    if declared != reflected {
        return Err(Error::registration(
            spec.name,
            format!(
                "declared bindings {declared:?} do not match the SPIR-V bindings {reflected:?} \
                 (set, binding, kind)"
            ),
        ));
    }
    Ok(entry.clone())
}

/// Compare the declaration against the device limits.
fn validate_against_limits(device: &Device, spec: &'static KernelSpec) -> Result<()> {
    let limits = device.limits();
    let workgroup = spec.workgroup_size;
    let invocations = u64::from(workgroup[0])
        .saturating_mul(u64::from(workgroup[1]))
        .saturating_mul(u64::from(workgroup[2]));
    if invocations > u64::from(limits.max_compute_work_group_invocations) {
        return Err(Error::registration(
            spec.name,
            format!(
                "workgroup size {workgroup:?} is {invocations} invocations, above \
                 maxComputeWorkGroupInvocations={} on \"{}\"",
                limits.max_compute_work_group_invocations,
                device.adapter().name
            ),
        ));
    }
    for axis in 0..3 {
        if workgroup[axis] > limits.max_compute_work_group_size[axis] {
            return Err(Error::registration(
                spec.name,
                format!(
                    "workgroup axis {axis} is {}, above maxComputeWorkGroupSize[{axis}]={} on \
                     \"{}\"",
                    workgroup[axis],
                    limits.max_compute_work_group_size[axis],
                    device.adapter().name
                ),
            ));
        }
    }
    if spec.push_constant_bytes > limits.max_push_constants_size {
        return Err(Error::registration(
            spec.name,
            format!(
                "{} push-constant bytes, above maxPushConstantsSize={} on \"{}\"",
                spec.push_constant_bytes,
                limits.max_push_constants_size,
                device.adapter().name
            ),
        ));
    }
    let storage = spec
        .bindings
        .iter()
        .filter(|binding| binding.kind == BindingKind::StorageBuffer)
        .count();
    if storage as u64 > u64::from(limits.max_descriptor_set_storage_buffers) {
        return Err(Error::registration(
            spec.name,
            format!(
                "{storage} storage buffers in one set, above \
                 maxDescriptorSetStorageBuffers={} on \"{}\"",
                limits.max_descriptor_set_storage_buffers,
                device.adapter().name
            ),
        ));
    }
    Ok(())
}

fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(Error::internal(format!(
            "SPIR-V has invalid length {} (must be a non-zero multiple of 4)",
            bytes.len()
        )));
    }
    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.as_chunks::<4>().0 {
        words.push(u32::from_le_bytes(*chunk));
    }
    if words[0] != 0x0723_0203 {
        return Err(Error::internal(format!(
            "SPIR-V has wrong magic 0x{:08x}",
            words[0]
        )));
    }
    Ok(words)
}

/// Compile and return a kernel, logging (not hiding) the pipeline-cache result.
pub fn create_kernel(device: &Arc<Device>, spec: &'static KernelSpec) -> Result<Arc<Kernel>> {
    let kernel = Kernel::new(device, spec)?;
    if device.instance().validation() {
        eprintln!(
            "[fn3vk] registered kernel \"{}\" (local_size={:?}, {} binding(s), {} push-constant bytes, shape: {})",
            spec.name,
            spec.workgroup_size,
            spec.bindings.len(),
            spec.push_constant_bytes,
            spec.shape
        );
    }
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_words_checks_magic_and_length() {
        assert!(spirv_words(&[]).is_err());
        assert!(spirv_words(&[0, 1, 2]).is_err());
        assert!(spirv_words(&[0u8; 8]).is_err());
        let words = spirv_words(super::super::kernels::smoke::SMOKE_ADD.spirv).unwrap();
        assert_eq!(words[0], 0x0723_0203);
    }
}
