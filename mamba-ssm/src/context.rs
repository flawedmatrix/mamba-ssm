use std::sync::{
    atomic::{AtomicBool, Ordering::Relaxed},
    Arc,
};

use candle::{DType, Device, Result, Shape, Tensor};
use candle_nn::VarMap;

/// Wrapper type for candle_nn::VarMap that supports prefixes
#[derive(Clone)]
pub struct Context {
    // Internally uses Arc<Mutex>
    vm: VarMap,
    path: Vec<String>,

    dtype: DType,
    device: Device,

    frozen: Arc<AtomicBool>,
}

impl Context {
    pub fn new(dtype: DType, device: &Device) -> Self {
        Self {
            vm: VarMap::new(),
            path: Vec::<String>::new(),
            dtype,
            device: device.clone(),
            frozen: Arc::new(false.into()),
        }
    }
    /// Returns a new `Context` adding `s` to the current prefix, like `cd`ing
    /// into a directory.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            vm: self.vm.clone(),
            path,
            dtype: self.dtype,
            device: self.device.clone(),
            frozen: self.frozen.clone(),
        }
    }
    pub fn set<K: AsRef<str>, V: AsRef<Tensor>>(&mut self, name: K, value: V) -> Result<()> {
        if self.frozen.load(Relaxed) {
            return Ok(());
        }
        let name = self.path(name.as_ref());
        let v_ref = value.as_ref();
        if v_ref.dtype() != self.dtype {
            candle::bail!(
                "cannot save context {name} because the dtype doesn't match: {} != {}",
                v_ref.dtype().as_str(),
                self.dtype.as_str()
            )
        }
        if !v_ref.device().same_device(self.device()) {
            candle::bail!(
                "cannot save context {name} because the device doesn't match: {:?} != {:?}",
                v_ref.device(),
                self.device()
            )
        }

        self.vm.set_one(name, value)
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    // Gets the variable, or initializes it with 0s if it isn't found
    pub fn get<S: Into<Shape>>(&self, shape: S, path: &str) -> Result<Tensor> {
        let name = self.path(path);
        let zero_init = candle_nn::Init::Const(0.);
        self.vm
            .get(shape, name.as_str(), zero_init, self.dtype, self.device())
    }
    /// This returns true only if a tensor with the passed in name is available. E.g. when passed
    /// `a`, true is returned if `prefix.a` exists but false is returned if only `prefix.a.b`
    /// exists.
    pub fn contains_tensor(&self, tensor_name: &str) -> bool {
        let path = self.path(tensor_name);

        let tensor_data = self.vm.data().lock().unwrap();
        tensor_data.contains_key(path.as_str())
    }

    /// The device used for variables.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The dtype used for variables.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Prevents setting variables on this context and all parent/child contexts
    pub fn freeze(&self) {
        self.frozen.store(true, Relaxed)
    }

    /// Allows setting variables on this context and all parent/child contexts
    pub fn unfreeze(&self) {
        self.frozen.store(false, Relaxed)
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").field("path", &self.path).finish()
    }
}
