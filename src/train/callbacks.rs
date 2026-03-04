use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TrainLogs {
    pub metrics: HashMap<String, f64>,
    pub epoch: usize,
    pub batch: usize,
    pub phase: String,
}

impl TrainLogs {
    pub fn new() -> Self {
        TrainLogs {
            metrics: HashMap::new(),
            epoch: 0,
            batch: 0,
            phase: "train".to_string(),
        }
    }
}

pub trait Callback: Send + Sync {
    fn on_epoch_begin(&mut self, epoch: usize, logs: &mut TrainLogs) {}
    fn on_epoch_end(&mut self, epoch: usize, logs: &mut TrainLogs) {}
    fn on_batch_begin(&mut self, batch: usize, logs: &mut TrainLogs) {}
    fn on_batch_end(&mut self, batch: usize, logs: &mut TrainLogs) {}
}

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
            if self.verbose {
                println!(
                    "Epoch {}: {} = {:.4}, saving checkpoint...",
                    epoch, self.monitor, value
                );
            }
        }
    }
}

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

        let improved = if let Some(best) = self.best_value {
            let delta = value - best;
            delta.abs() > self.min_delta
                && ((self.monitor.contains("loss") && value < best)
                    || (!self.monitor.contains("loss") && value > best))
        } else {
            true
        };

        if improved {
            self.best_value = Some(value);
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                println!("Early stopping triggered at epoch {}!", epoch);
            }
        }
    }
}

pub struct LearningRateScheduler {
    schedule: String,
    lr: f64,
    step_size: usize,
    gamma: f64,
    T_max: usize,
    eta_min: f64,
}

impl LearningRateScheduler {
    pub fn new(
        schedule: String,
        lr: f64,
        step_size: usize,
        gamma: f64,
        T_max: usize,
        eta_min: f64,
    ) -> Self {
        LearningRateScheduler {
            schedule,
            lr,
            step_size,
            gamma,
            T_max,
            eta_min,
        }
    }
}

impl Callback for LearningRateScheduler {
    fn on_epoch_end(&mut self, epoch: usize, logs: &mut TrainLogs) {
        let new_lr = match self.schedule.as_str() {
            "step" => self.lr * self.gamma.powf((epoch / self.step_size) as f64),
            "cosine" => {
                self.eta_min
                    + 0.5
                        * (self.lr - self.eta_min)
                        * (1.0 + ((epoch as f64 * std::f64::consts::PI) / self.T_max as f64).cos())
            }
            _ => self.lr,
        };

        logs.metrics.insert("lr".to_string(), new_lr);
    }
}

pub struct CSVLogger {
    filepath: String,
    header_written: bool,
    verbose: bool,
}

impl CSVLogger {
    pub fn new(filepath: String, verbose: bool) -> Self {
        CSVLogger {
            filepath,
            header_written: false,
            verbose,
        }
    }
}

impl Callback for CSVLogger {
    fn on_epoch_end(&mut self, epoch: usize, logs: &mut TrainLogs) {
        if self.verbose {
            println!("Epoch {} metrics: {:?}", epoch, logs.metrics);
        }
    }
}

pub struct ProgressBar {
    verbose: bool,
}

impl ProgressBar {
    pub fn new() -> Self {
        ProgressBar { verbose: true }
    }
}

impl Callback for ProgressBar {
    fn on_batch_end(&mut self, batch: usize, logs: &mut TrainLogs) {
        if self.verbose && batch % 10 == 0 {
            let loss = logs
                .metrics
                .get("loss")
                .map(|l| format!("{:.4}", l))
                .unwrap_or_default();
            print!("\rBatch {}: loss = {}", batch, loss);
        }
    }
}
