#![allow(dead_code)]
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TrainLogs {
    pub metrics: HashMap<String, f64>,
    pub epoch: usize,
    pub batch: usize,
    pub phase: String,
    pub should_stop: bool,
}

impl TrainLogs {
    pub fn new() -> Self {
        TrainLogs {
            metrics: HashMap::new(),
            epoch: 0,
            batch: 0,
            phase: "train".to_string(),
            should_stop: false,
        }
    }
}

impl Default for TrainLogs {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
pub trait Callback: Send + Sync {
    fn on_epoch_begin(&mut self, _epoch: usize, _logs: &mut TrainLogs) {}
    fn on_epoch_end(&mut self, _epoch: usize, _logs: &mut TrainLogs) {}
    fn on_batch_begin(&mut self, _batch: usize, _logs: &mut TrainLogs) {}
    fn on_batch_end(&mut self, _batch: usize, _logs: &mut TrainLogs) {}
}

#[allow(dead_code)]
pub struct ModelCheckpoint {
    dir_path: String,
    monitor: String,
    mode: String,
    save_best_only: bool,
    verbose: bool,
    best_value: Option<f64>,
}

impl ModelCheckpoint {
    pub fn new(
        dir_path: String,
        monitor: String,
        mode: String,
        save_best_only: bool,
        verbose: bool,
    ) -> Self {
        ModelCheckpoint {
            dir_path,
            monitor,
            mode,
            save_best_only,
            verbose,
            best_value: None,
        }
    }
}

impl Callback for ModelCheckpoint {
    fn on_epoch_end(&mut self, epoch: usize, logs: &mut TrainLogs) {
        let value = if let Some(v) = logs.metrics.get(&self.monitor) {
            *v
        } else {
            return;
        };

        let is_best = match self.mode.as_str() {
            "min" => self.best_value.map(|b| value < b).unwrap_or(true),
            "max" => self.best_value.map(|b| value > b).unwrap_or(true),
            _ => false,
        };

        if is_best || !self.save_best_only {
            self.best_value = Some(value);
            // Create directory if it doesn't exist
            let dir = Path::new(&self.dir_path);
            if !dir.exists() {
                let _ = fs::create_dir_all(dir);
            }
            let filename = if self.save_best_only {
                "best_model.fnn".to_string()
            } else {
                format!("checkpoint_epoch_{}.fnn", epoch)
            };
            let filepath = dir.join(&filename);
            if self.verbose {
                println!(
                    "Epoch {}: {} = {:.4}, saving checkpoint to {}",
                    epoch,
                    self.monitor,
                    value,
                    filepath.display()
                );
            }
            // Note: actual model saving is handled by the training loop
            // which has access to the model. The callback stores the path.
            logs.metrics.insert("checkpoint_path".to_string(), value);
        }
    }
}

#[allow(dead_code)]
pub struct EarlyStopping {
    monitor: String,
    patience: usize,
    min_delta: f64,
    restore_best_weights: bool,
    counter: usize,
    best_value: Option<f64>,
}

impl EarlyStopping {
    pub fn new(
        monitor: String,
        patience: usize,
        min_delta: f64,
        restore_best_weights: bool,
    ) -> Self {
        EarlyStopping {
            monitor,
            patience,
            min_delta,
            restore_best_weights,
            counter: 0,
            best_value: None,
        }
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(&mut self, epoch: usize, logs: &mut TrainLogs) {
        let value = if let Some(v) = logs.metrics.get(&self.monitor) {
            *v
        } else {
            return;
        };

        let is_improved = match self.best_value {
            None => true,
            Some(best) => value < best - self.min_delta,
        };

        if is_improved {
            self.best_value = Some(value);
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                if self.restore_best_weights {
                    println!(
                        "Epoch {}: Early stopping triggered. Restoring best weights.",
                        epoch
                    );
                } else {
                    println!("Epoch {}: Early stopping triggered.", epoch);
                }
                logs.should_stop = true;
            }
        }
    }
}

#[allow(dead_code)]
pub struct LearningRateScheduler {
    schedule_type: String,
    initial_lr: f64,
    decay_rate: f64,
    decay_steps: usize,
    min_lr: f64,
    total_epochs: usize,
}

impl LearningRateScheduler {
    pub fn new(
        schedule_type: String,
        initial_lr: f64,
        decay_rate: f64,
        decay_steps: usize,
        min_lr: f64,
        total_epochs: usize,
    ) -> Self {
        LearningRateScheduler {
            schedule_type,
            initial_lr,
            decay_rate,
            decay_steps,
            min_lr,
            total_epochs,
        }
    }

    pub fn get_lr(&self, epoch: usize) -> f64 {
        match self.schedule_type.as_str() {
            "step" => {
                let factor = self.decay_rate.powi((epoch / self.decay_steps) as i32);
                (self.initial_lr * factor).max(self.min_lr)
            }
            "cosine" => {
                let progress = epoch as f64 / self.total_epochs as f64;
                let lr = self.initial_lr * 0.5 * (1.0 + std::f64::consts::PI * progress).cos();
                lr.max(self.min_lr)
            }
            "exponential" => {
                let lr = self.initial_lr * self.decay_rate.powi(epoch as i32);
                lr.max(self.min_lr)
            }
            _ => self.initial_lr,
        }
    }
}

impl Callback for LearningRateScheduler {
    fn on_epoch_begin(&mut self, epoch: usize, logs: &mut TrainLogs) {
        let lr = self.get_lr(epoch);
        logs.metrics.insert("lr".to_string(), lr);
    }
}

#[allow(dead_code)]
pub struct CSVLogger {
    filepath: String,
    fields: Option<Vec<String>>,
    epoch: usize,
    initialized: bool,
}

impl CSVLogger {
    pub fn new(filepath: String, fields: Option<Vec<String>>) -> Self {
        CSVLogger {
            filepath,
            fields,
            epoch: 0,
            initialized: false,
        }
    }
}

impl Callback for CSVLogger {
    fn on_epoch_begin(&mut self, epoch: usize, _logs: &mut TrainLogs) {
        self.epoch = epoch;
        if !self.initialized {
            // Create parent directory if needed
            if let Some(parent) = Path::new(&self.filepath).parent() {
                if !parent.exists() {
                    let _ = fs::create_dir_all(parent);
                }
            }
            let _ = fs::write(&self.filepath, "");
            self.initialized = true;
        }
    }

    fn on_epoch_end(&mut self, epoch: usize, logs: &mut TrainLogs) {
        if !self.initialized {
            return;
        }

        let field_order = vec![
            "epoch".to_string(),
            "loss".to_string(),
            "val_loss".to_string(),
            "accuracy".to_string(),
            "val_accuracy".to_string(),
            "lr".to_string(),
        ];
        let fields = if let Some(ref f) = self.fields {
            f.clone()
        } else {
            let mut fields = field_order.clone();
            let extra_keys: Vec<String> = logs
                .metrics
                .keys()
                .filter(|k| !fields.contains(k))
                .cloned()
                .collect();
            fields.extend(extra_keys);
            fields
        };

        if epoch == 0 {
            let header = fields.join(",");
            let _ = fs::write(&self.filepath, format!("{}\n", header));
        }

        let values: Vec<String> = fields
            .iter()
            .map(|f| {
                if f == "epoch" {
                    epoch.to_string()
                } else {
                    logs.metrics
                        .get(f)
                        .map(|v| v.to_string())
                        .unwrap_or_default()
                }
            })
            .collect();

        let line = format!("{}\n", values.join(","));
        if let Ok(mut f) = fs::OpenOptions::new().append(true).open(&self.filepath) {
            use std::io::Write;
            let _ = f.write_all(line.as_bytes());
        }
    }
}
