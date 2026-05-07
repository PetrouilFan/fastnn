pub struct Conv2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub has_bias: bool,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
    pub edges: Vec<Edge>,
    pub inputs: Vec<Tensor>,
    // Pre-allocated scalar constants to avoid per-call allocation
    stride_scalar: Tensor,
    padding_scalar: Tensor,
    dilation_scalar: Tensor,
    groups_scalar: Tensor,
    zero_scalar: Tensor,
    one_scalar: Tensor,
}

impl Conv2dBackward {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: Tensor,
        weight: Tensor,
        has_bias: bool,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        edges: Vec<Edge>,
    ) -> Self {
        let inputs = vec![input.clone(), weight.clone()];
        Conv2dBackward {
            input,
            weight,
            has_bias,
            stride,
            padding,
            dilation,
            groups,
            edges,
            inputs,
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            dilation_scalar: Tensor::from_scalar(dilation as f32),
            groups_scalar: Tensor::from_scalar(groups as f32),
            zero_scalar: Tensor::from_scalar(0.0),
            one_scalar: Tensor::from_scalar(1.0),
        }
    }
}

impl Node for Conv2dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad = crate::autograd::extract_first_grad(grad_outputs);

        let input = &self.input;
        let weight = &self.weight;

        let weight_shape = weight.shape();
        let out_channels = weight_shape[0];
        let in_channels_per_group = weight_shape[1];
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_h = input_shape[2];
        let in_w = input_shape[3];

        let grad_shape = grad.shape();
        let out_h = grad_shape[2];
        let out_w = grad_shape[3];

        let stride = self.stride;
        let padding = self.padding;
        let dilation = self.dilation;
        let groups = self.groups;

        // Compute grad_input: convolve grad_output with rotated weight
        // Rotate weight by 180 degrees (flip both spatial dimensions)
        let weight_rotated = weight.flip(2).flip(3);

        // For grad_input, we need to do a transposed convolution:
        // - Output padding is determined by stride and kernel size
        // - Padding for grad_input = dilation * (kernel - 1) - padding
        let grad_input_padding_h = dilation * (kernel_h - 1) - padding;
        let _grad_input_padding_w = dilation * (kernel_w - 1) - padding;

        // Use pre-allocated scalars
        let padding_scalar = &self.padding_scalar;
        let dilation_scalar = &self.dilation_scalar;
        let zero_scalar = &self.zero_scalar;
        let one_scalar = &self.one_scalar;

        // OPTIMIZATION: grad_input computation
        let grad_input = if groups == 1 {
            // Standard convolution backward
            if stride == 1 {
                // Use conv2d with adjusted parameters
                dispatch(
                    "conv2d",
                    crate::dispatcher::DispatchKey::Cpu,
                    &[
                        &grad,
                        &weight_rotated,
                        zero_scalar,
                        one_scalar,
                        padding_scalar,
                        dilation_scalar,
                        one_scalar,
                    ],
                )[0]
                .clone()
            } else {
                // For stride > 1, dilate grad_output first
                // Get CPU copy of grad_output (avoid unnecessary copy if already CPU)
                let grad_cpu = if grad.inner.is_cpu() {
                    None
                } else {
                    Some(grad.to_cpu())
                };
                let grad_data = if let Some(ref t) = grad_cpu {
                    t.data_ptr_f32()
                } else {
                    grad.data_ptr_f32()
                };
                let grad_h = out_h;
                let grad_w = out_w;
                let dilated_h = grad_h + (grad_h - 1) * (stride - 1);
                let dilated_w = grad_w + (grad_w - 1) * (stride - 1);

                // Use zeros to ensure inserted stride gaps are zero-initialized
                let mut dilated_grad = Tensor::zeros(
                    vec![batch_size, out_channels, dilated_h, dilated_w],
                    grad.dtype(),
                    grad.device(),
                );
                let dilated_data = dilated_grad.data_ptr_f32_mut();

                // Fill dilated grad with (stride-1) zero rows/cols inserted
                for b in 0..batch_size {
                    for c in 0..out_channels {
                        for oh in 0..grad_h {
                            for ow in 0..grad_w {
                                let src_idx = (b * out_channels + c) * grad_h * grad_w
                                    + oh * grad_w
                                    + ow;
                                let dst_idx = (b * out_channels + c) * dilated_h * dilated_w
                                    + (oh * stride) * dilated_w
                                    + (ow * stride);
                                unsafe {
                                    *dilated_data.add(dst_idx as usize) = *grad_data.add(src_idx as usize);
                                }
                            }
                        }
                    }
                }
                // grad_cpu (if any) gets dropped here when it goes out of scope

                dispatch(
                    "conv2d",
                    crate::dispatcher::DispatchKey::Cpu,
                    &[
                        &dilated_grad,
                        &weight_rotated,
                        zero_scalar,
                        one_scalar,
                        padding_scalar,
                        dilation_scalar,
                        one_scalar,
                    ],
                )[0]
                .clone()
            }
        } else {
            // Grouped convolution: process each group separately
            let mut grad_input_parts = Vec::new();
            let out_channels_per_group = out_channels / groups;
            let in_channels_actual = in_channels_per_group;

            for g in 0..groups {
                let g = g as usize;
                let out_c_start = g * out_channels_per_group as usize;
                let out_c_end = (g + 1) * out_channels_per_group as usize;
                let in_c_start = g * in_channels_actual as usize;
                let in_c_end = (g + 1) * in_channels_actual as usize;

                // Slice grad_output for this group
                let grad_g = grad.slice(1, out_c_start as i64, out_c_end as i64, 1);
                // Slice weight for this group
                let weight_g = weight.slice(0, out_c_start as i64, out_c_end as i64, 1);
                let weight_g_rotated = weight_g.flip(2).flip(3);
                // Slice input for this group
                let input_g = input.slice(1, in_c_start as i64, in_c_end as i64, 1);

                let _input_g_shape = input_g.shape();
                let _in_h_g = input_g.shape()[2];
                let _in_w_g = input_g.shape()[3];

                let grad_input_g = if stride == 1 {
                    dispatch(
                        "conv2d",
                        crate::dispatcher::DispatchKey::Cpu,
                        &[
                            &grad_g,
                            &weight_g_rotated,
                            zero_scalar,
                            one_scalar,
                            &Tensor::from_scalar(grad_input_padding_h as f32),
                            dilation_scalar,
                            one_scalar,
                        ],
                    )[0]
                    .clone()
                } else {
                    let out_h_g = grad_g.shape()[2];
                    let out_w_g = grad_g.shape()[3];
                    let dilated_h = out_h_g + (out_h_g - 1) * (stride - 1);
                    let dilated_w = out_w_g + (out_w_g - 1) * (stride - 1);

                    let mut dilated_grad_data = vec![
                        0.0f32;
                        batch_size as usize
                            * out_channels_per_group as usize
                            * dilated_h as usize
                            * dilated_w as usize
                    ];
                    let grad_g_cpu = if grad_g.inner.is_cpu() {
                        grad_g
                    } else {
                        grad_g.to_cpu()
                    };
                    let grad_g_data = grad_g_cpu.as_f32_slice();

                    for b in 0..batch_size as usize {
                        for c in 0..out_channels_per_group as usize {
                            for oh in 0..out_h_g as usize {
                                for ow in 0..out_w_g as usize {
                                    let src_idx = b
                                        * (out_channels_per_group as usize
                                            * out_h_g as usize
                                            * out_w_g as usize)
                                        + c * (out_h_g as usize * out_w_g as usize)
                                        + oh * out_w_g as usize
                                        + ow;
                                    let dst_idx = b
                                        * (out_channels_per_group as usize
                                            * dilated_h as usize
                                            * dilated_w as usize)
                                        + c * (dilated_h as usize * dilated_w as usize)
                                        + (oh * stride as usize) * dilated_w as usize
                                        + (ow * stride as usize);
                                    dilated_grad_data[dst_idx] = grad_g_data[src_idx];
                                }
                            }
                        }
                    }

                    let dilated_grad = Tensor::from_vec(
                        dilated_grad_data,
                        vec![batch_size, out_channels_per_group, dilated_h, dilated_w],
                    );

                    dispatch(
                        "conv2d",
                        crate::dispatcher::DispatchKey::Cpu,
                        &[
                            &dilated_grad,
                            &weight_g_rotated,
                            zero_scalar,
                            one_scalar,
                            &Tensor::from_scalar(grad_input_padding_h as f32),
                            dilation_scalar,
                            one_scalar,
                        ],
                    )[0]
                    .clone()
                };

                grad_input_parts.push(grad_input_g);
            }

            // Concatenate along channel dimension
            if grad_input_parts.len() == 1 {
                grad_input_parts[0].clone()
            } else {
                Tensor::cat(&grad_input_parts, 1)
            }
        };

        // Compute grad_weight: convolve input with grad_output
        // For each output channel, grad_weight[c] = sum over batch of conv2d(input, grad_output[c])
        let grad_weight = if groups == 1 {
            // Reshape for matrix multiplication approach
            // im2col on input: [batch, in_channels * kernel_h * kernel_w, out_h * out_w]
            // grad_output reshaped: [batch, out_channels, out_h * out_w]
            // grad_weight = grad_output @ im2col(input)^T / batch

            // Simplified: use the fact that grad_weight[c, ic, kh, kw] = sum over b,h,w of input[b, ic, h*stride+kh, w*stride+kw] * grad[b, c, h, w]
            let mut grad_weight_data =
                vec![0.0f32; (out_channels * in_channels * kernel_h * kernel_w) as usize];

            let input_cpu = if input.inner.is_cpu() {
                input.clone()
            } else {
                input.to_cpu()
            };
            let grad_cpu = if grad.inner.is_cpu() {
                grad.clone()
            } else {
                grad.to_cpu()
            };
            let input_data = input_cpu.as_f32_slice();
            let grad_data = grad_cpu.as_f32_slice();

            for b in 0..batch_size as usize {
                for oc in 0..out_channels as usize {
                    for ic in 0..in_channels as usize {
                        for kh in 0..kernel_h as usize {
                            for kw in 0..kernel_w as usize {
                                let mut sum = 0.0f32;
                                for oh in 0..out_h as usize {
                                    for ow in 0..out_w as usize {
                                        let ih = oh * stride as usize + kh * dilation as usize;
                                        let iw = ow * stride as usize + kw * dilation as usize;

                                        if ih < in_h as usize && iw < in_w as usize {
                                            let input_idx = b
                                                * (in_channels * in_h * in_w) as usize
                                                + ic * (in_h * in_w) as usize
                                                + ih * in_w as usize
                                                + iw;
                                            let grad_idx = b
                                                * (out_channels * out_h * out_w) as usize
                                                + oc * (out_h * out_w) as usize
                                                + oh * out_w as usize
                                                + ow;
                                            sum += input_data[input_idx] * grad_data[grad_idx];
                                        }
                                    }
                                }
                                let gw_idx = oc * (in_channels * kernel_h * kernel_w) as usize
                                    + ic * (kernel_h * kernel_w) as usize
                                    + kh * kernel_w as usize
                                    + kw;
                                grad_weight_data[gw_idx] += sum;
                            }
                        }
                    }
                }
            }

            Tensor::from_vec(
                grad_weight_data,
                vec![out_channels, in_channels, kernel_h, kernel_w],
            )
        } else {
            // Grouped: compute per group
            let out_channels_per_group = out_channels / groups;
            let in_channels_per_group_actual = in_channels_per_group;
            let mut grad_weight_data = vec![
                0.0f32;
                (out_channels * in_channels_per_group_actual * kernel_h * kernel_w)
                    as usize
            ];

            let input_cpu = if input.inner.is_cpu() {
                input.clone()
            } else {
                input.to_cpu()
            };
            let grad_cpu = if grad.inner.is_cpu() {
                grad.clone()
            } else {
                grad.to_cpu()
            };
            let input_data = input_cpu.as_f32_slice();
            let grad_data = grad_cpu.as_f32_slice();

            for g in 0..groups as usize {
                let oc_start = g * out_channels_per_group as usize;
                let oc_end = (g + 1) * out_channels_per_group as usize;
                let ic_start = g * in_channels_per_group_actual as usize;
                let ic_end = (g + 1) * in_channels_per_group_actual as usize;

                for b in 0..batch_size as usize {
                    for oc in oc_start..oc_end {
                        for ic in ic_start..ic_end {
                            for kh in 0..kernel_h as usize {
                                for kw in 0..kernel_w as usize {
                                    let mut sum = 0.0f32;
                                    for oh in 0..out_h as usize {
                                        for ow in 0..out_w as usize {
                                            let ih = oh * stride as usize + kh * dilation as usize;
                                            let iw = ow * stride as usize + kw * dilation as usize;

                                            if ih < in_h as usize && iw < in_w as usize {
                                                let input_idx = b
                                                    * (in_channels * in_h * in_w) as usize
                                                    + ic * (in_h * in_w) as usize
                                                    + ih * in_w as usize
                                                    + iw;
                                                let grad_idx = b
                                                    * (out_channels * out_h * out_w) as usize
                                                    + oc * (out_h * out_w) as usize
                                                    + oh * out_w as usize
                                                    + ow;
                                                sum += input_data[input_idx] * grad_data[grad_idx];
                                            }
                                        }
                                    }
                                    let ic_local = ic - ic_start;
                                    let _oc_local = oc - oc_start;
                                    let gw_idx = oc
                                        * (in_channels_per_group_actual * kernel_h * kernel_w)
                                            as usize
                                        + ic_local * (kernel_h * kernel_w) as usize
                                        + kh * kernel_w as usize
                                        + kw;
                                    grad_weight_data[gw_idx] += sum;
                                }
                            }
                        }
                    }
                }
            }

            Tensor::from_vec(
                grad_weight_data,
                vec![
                    out_channels,
                    in_channels_per_group_actual,
                    kernel_h,
                    kernel_w,
                ],
            )
        };

        // Compute grad_bias: sum grad_output over batch and spatial dimensions
        let grad_bias = if self.has_bias {
            let mut grad_bias_data = vec![0.0f32; out_channels as usize];
            let grad_cpu = if grad.inner.is_cpu() {
                grad.clone()
            } else {
                grad.to_cpu()
            };
            let grad_data = grad_cpu.as_f32_slice();

            for b in 0..batch_size as usize {
                for oc in 0..out_channels as usize {
                    for oh in 0..out_h as usize {
                        for ow in 0..out_w as usize {
                            let idx = b * (out_channels * out_h * out_w) as usize
                                + oc * (out_h * out_w) as usize
                                + oh * out_w as usize
                                + ow;
                            grad_bias_data[oc] += grad_data[idx];
                        }
                    }
                }
            }

            Some(Tensor::from_vec(grad_bias_data, vec![out_channels]))
        } else {
            None
        };

        vec![Some(grad_input), Some(grad_weight), grad_bias]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "Conv2dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[allow(dead_code)]
pub struct ConvTranspose2dBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: i64,
    pub padding: i64,
    pub kernel_size: i64,
    pub edges: Vec<Edge>,
    // Pre-allocated scalar constants
    stride_scalar: Tensor,
    padding_scalar: Tensor,
    zero_scalar: Tensor,
    one_scalar: Tensor,
}

impl ConvTranspose2dBackward {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
        stride: i64,
        padding: i64,
        kernel_size: i64,
        edges: Vec<Edge>,
    ) -> Self {
        ConvTranspose2dBackward {
            input,
            weight,
            bias,
            stride,
            padding,
            kernel_size,
            edges,
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            zero_scalar: Tensor::from_scalar(0.0),
            one_scalar: Tensor::from_scalar(1.0),
        }
    }
}

impl Node for ConvTranspose2dBackward {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        let grad_output = crate::autograd::extract_first_grad(grad_outputs);

        let in_shape = self.input.shape();
        let _batch = in_shape[0];
        let _in_channels = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];
        let _out_channels = self.weight.shape()[0];
        let kernel_h = self.weight.shape()[2];
        let kernel_w = self.weight.shape()[3];

        let _h_out = (h_in - 1) * self.stride - 2 * self.padding + kernel_h;
        let _w_out = (w_in - 1) * self.stride - 2 * self.padding + kernel_w;

        // grad_input: conv2d(grad_output, weight, stride=1, padding=kernel_size-1-padding)
        let pad_input = kernel_h - 1 - self.padding;
        let grad_input = dispatch(
            "conv2d",
            crate::dispatcher::DispatchKey::Cpu,
            &[
                &grad_output,
                &self.weight,
                &self.one_scalar,
                &Tensor::from_scalar(pad_input as f32),
                &self.zero_scalar,
                &self.one_scalar,
            ],
        )[0]
        .clone();

        // grad_weight: conv2d(input, grad_output) with appropriate padding
        // Reshape for im2col-style: input [B, C_in, H_in, W_in], grad_output [B, C_out, H_out, W_out]
        // grad_weight [C_in, C_out, kH, kW]
        let grad_weight = dispatch(
            "conv2d",
            crate::dispatcher::DispatchKey::Cpu,
            &[
                &self.input,
                &grad_output,
                &self.one_scalar,
                &self.padding_scalar,
                &self.zero_scalar,
                &self.one_scalar,
            ],
        )[0]
        .clone();

        // grad_bias: sum over batch, height, width dimensions
        let grad_bias = if self.bias.is_some() {
            Some(grad_output.sum(0, false).sum(1, false).sum(1, false))
        } else {
            None
        };

        let mut grads = vec![Some(grad_input), Some(grad_weight)];
        if grad_bias.is_some() {
            grads.push(grad_bias);
        }
        grads
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        if self.bias.is_some() {
            3
        } else {
            2
        }
    }

    fn name(&self) -> &str {
        "ConvTranspose2dBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        std::slice::from_ref(&self.input)
    }
}

