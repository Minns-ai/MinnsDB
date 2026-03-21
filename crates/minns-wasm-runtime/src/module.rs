//! ModuleInstance: a compiled and instantiated WASM module ready to call.

use std::sync::Arc;

use tokio::sync::RwLock;
use wasmtime::{Instance, Linker, Module, Store};

use agent_db_tables::catalog::TableCatalog;

use crate::abi::{self, ModuleDescriptor};
use crate::error::WasmError;
use crate::host_functions::{self, HostEnv};
use crate::permissions::PermissionSet;
use crate::runtime::{WasmRuntime, EPOCH_TICKS_PER_SECOND};
use crate::usage::ModuleUsageCounters;

/// A loaded WASM module instance with its environment.
pub struct ModuleInstance {
    /// Compiled module (reusable across instantiations).
    compiled: Module,
    /// Module descriptor (cached from __minns_describe__).
    pub descriptor: ModuleDescriptor,
    /// Content-addressed hash of the WASM bytes.
    pub blob_hash: [u8; 32],
    /// Permission set granted to this module.
    pub permissions: PermissionSet,
    /// Usage counters.
    pub usage: Arc<ModuleUsageCounters>,
    /// Group ID for this module's operations.
    pub group_id: u64,
    /// Module ID (assigned by registry).
    pub module_id: u64,
    /// Whether the module is enabled.
    pub enabled: bool,
    /// Shared references needed for creating stores.
    table_catalog: Arc<RwLock<TableCatalog>>,
    /// Life budget per call.
    life_budget: u64,
}

impl ModuleInstance {
    /// Compile and instantiate a module from WASM bytes.
    /// Calls __minns_describe__ to extract the module descriptor.
    pub fn load(
        runtime: &WasmRuntime,
        wasm_bytes: &[u8],
        permissions: PermissionSet,
        group_id: u64,
        module_id: u64,
        table_catalog: Arc<RwLock<TableCatalog>>,
    ) -> Result<Self, WasmError> {
        let blob_hash = WasmRuntime::hash_module(wasm_bytes);
        let compiled = runtime.compile(wasm_bytes)?;
        let usage = Arc::new(ModuleUsageCounters::new());
        let life_budget = runtime.config().default_life_budget;

        let mut instance = ModuleInstance {
            compiled,
            descriptor: ModuleDescriptor {
                name: String::new(),
                version: String::new(),
                functions: Vec::new(),
                triggers: Vec::new(),
                permissions: Vec::new(),
            },
            blob_hash,
            permissions,
            usage,
            group_id,
            module_id,
            enabled: true,
            table_catalog,
            life_budget,
        };

        // Extract descriptor by calling __minns_describe__
        instance.descriptor = instance.call_describe(runtime)?;

        // Call __minns_init__ if exported
        instance.call_init(runtime)?;

        Ok(instance)
    }

    /// Create a fresh Store + Instance for a single call.
    fn create_store_and_instance(
        &self,
        runtime: &WasmRuntime,
    ) -> Result<(Store<HostEnv>, Instance), WasmError> {
        let env = HostEnv::new(
            self.permissions.clone(),
            self.usage.clone(),
            self.group_id,
            self.module_id,
            self.table_catalog.clone(),
        );

        let mut store = Store::new(runtime.engine(), env);
        store.limiter(|data| &mut data.limiter);

        // Set life budget and epoch deadline
        store
            .set_fuel(self.life_budget)
            .map_err(|e| WasmError::ExecutionError(format!("set life budget: {}", e)))?;
        store.set_epoch_deadline(EPOCH_TICKS_PER_SECOND * runtime.config().wall_time_limit_secs);

        // Link host functions
        let mut linker = Linker::new(runtime.engine());
        host_functions::register_host_functions(&mut linker)?;

        let instance = linker
            .instantiate(&mut store, &self.compiled)
            .map_err(|e| WasmError::InstantiationError(e.to_string()))?;

        Ok((store, instance))
    }

    /// Call __minns_describe__ to extract the module descriptor.
    ///
    /// ABI: __minns_describe__() returns a pointer to MessagePack bytes in WASM memory.
    /// __minns_describe_len__() returns the length. If describe_len is not exported,
    /// falls back to reading from the host env's last_result buffer.
    fn call_describe(&self, runtime: &WasmRuntime) -> Result<ModuleDescriptor, WasmError> {
        let (mut store, instance) = self.create_store_and_instance(runtime)?;

        let describe_fn = instance
            .get_typed_func::<(), i32>(&mut store, "__minns_describe__")
            .map_err(|_| WasmError::InvalidModule("missing __minns_describe__ export".into()))?;

        let result_ptr = describe_fn.call(&mut store, ()).map_err(WasmError::from)?;

        if result_ptr < 0 {
            return Err(WasmError::InvalidModule(
                "__minns_describe__ returned error".into(),
            ));
        }

        // Try to get the length from __minns_describe_len__ export
        let result_bytes = if let Ok(len_fn) =
            instance.get_typed_func::<(), i32>(&mut store, "__minns_describe_len__")
        {
            let len = len_fn.call(&mut store, ()).map_err(WasmError::from)?;
            if len > 0 && result_ptr > 0 {
                // Read directly from WASM memory
                let memory = instance
                    .get_memory(&mut store, "memory")
                    .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;
                abi::read_from_wasm(&memory, &store, result_ptr, len)?
            } else {
                store.data().last_result.clone()
            }
        } else {
            // Fallback: read from host env's last_result buffer
            store.data().last_result.clone()
        };

        if result_bytes.is_empty() {
            return Err(WasmError::InvalidModule(
                "__minns_describe__ returned empty descriptor. The module must export \
                 __minns_describe__() returning a pointer to MessagePack bytes and \
                 __minns_describe_len__() returning the byte length."
                    .into(),
            ));
        }

        abi::from_msgpack(&result_bytes)
    }

    /// Call __minns_init__ if the module exports it.
    fn call_init(&self, runtime: &WasmRuntime) -> Result<(), WasmError> {
        let (mut store, instance) = self.create_store_and_instance(runtime)?;

        // __minns_init__ is optional
        let init_fn = match instance.get_typed_func::<(), i32>(&mut store, "__minns_init__") {
            Ok(f) => f,
            Err(_) => return Ok(()), // no init function, that's fine
        };

        let result_code = init_fn.call(&mut store, ()).map_err(WasmError::from)?;

        if result_code != 0 {
            return Err(WasmError::ExecutionError(format!(
                "__minns_init__ returned error code {}",
                result_code
            )));
        }

        Ok(())
    }

    /// Call a named function on the module.
    /// func_name is looked up in the descriptor to get the func_id.
    /// args are MessagePack bytes passed to the module.
    /// Returns MessagePack result bytes.
    pub fn call_function(
        &self,
        runtime: &WasmRuntime,
        func_name: &str,
        args: &[u8],
    ) -> Result<Vec<u8>, WasmError> {
        if !self.enabled {
            return Err(WasmError::ExecutionError("module is disabled".into()));
        }

        // Look up function ID from descriptor
        let func_id = self
            .descriptor
            .functions
            .iter()
            .position(|f| f.name == func_name)
            .ok_or_else(|| {
                WasmError::ExecutionError(format!("function '{}' not found in module", func_name))
            })? as i32;

        let (mut store, instance) = self.create_store_and_instance(runtime)?;

        let call_fn = instance
            .get_typed_func::<(i32, i32, i32), i32>(&mut store, "__minns_call__")
            .map_err(|_| WasmError::InvalidModule("missing __minns_call__ export".into()))?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

        // Write args into WASM memory
        let alloc_fn = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc")
            .map_err(|_| WasmError::AbiError("no alloc export".into()))?;

        let (args_ptr, args_len) = abi::write_to_wasm(&memory, &mut store, &alloc_fn, args)?;

        // Record the call
        self.usage.record_call();

        // Call the function
        let life_before = store.get_fuel().unwrap_or(0);
        let result_code = call_fn
            .call(&mut store, (func_id, args_ptr, args_len))
            .map_err(WasmError::from)?;

        // Record life consumed
        let life_after = store.get_fuel().unwrap_or(0);
        let life_consumed = life_before.saturating_sub(life_after);
        self.usage.record_life(life_consumed);

        // Read result
        let result_bytes = store.data().last_result.clone();

        if result_code < 0 {
            let err_msg = if result_bytes.is_empty() {
                format!(
                    "function '{}' returned error code {}",
                    func_name, result_code
                )
            } else {
                abi::from_msgpack::<String>(&result_bytes).unwrap_or_else(|_| {
                    format!(
                        "function '{}' returned error code {}",
                        func_name, result_code
                    )
                })
            };
            return Err(WasmError::ExecutionError(err_msg));
        }

        Ok(result_bytes)
    }

    /// Call a trigger handler on the module.
    pub fn call_trigger(
        &self,
        runtime: &WasmRuntime,
        trigger_id: i32,
        event: &[u8],
    ) -> Result<(), WasmError> {
        if !self.enabled {
            return Ok(()); // silently skip disabled modules
        }

        let (mut store, instance) = self.create_store_and_instance(runtime)?;

        let trigger_fn = match instance
            .get_typed_func::<(i32, i32, i32), i32>(&mut store, "__minns_on_trigger__")
        {
            Ok(f) => f,
            Err(_) => return Ok(()), // no trigger handler
        };

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| WasmError::AbiError("no memory export".into()))?;

        let alloc_fn = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc")
            .map_err(|_| WasmError::AbiError("no alloc export".into()))?;

        let (event_ptr, event_len) = abi::write_to_wasm(&memory, &mut store, &alloc_fn, event)?;

        self.usage.record_call();

        let life_before = store.get_fuel().unwrap_or(0);
        let result_code = trigger_fn
            .call(&mut store, (trigger_id, event_ptr, event_len))
            .map_err(WasmError::from)?;

        let life_after = store.get_fuel().unwrap_or(0);
        self.usage
            .record_life(life_before.saturating_sub(life_after));

        if result_code < 0 {
            tracing::warn!(
                "trigger {} on module '{}' returned error code {}",
                trigger_id,
                self.descriptor.name,
                result_code
            );
        }

        Ok(())
    }

    /// Call a schedule handler on the module.
    pub fn call_schedule(&self, runtime: &WasmRuntime, schedule_id: i32) -> Result<(), WasmError> {
        if !self.enabled {
            return Ok(());
        }

        let (mut store, instance) = self.create_store_and_instance(runtime)?;

        let schedule_fn =
            match instance.get_typed_func::<i32, i32>(&mut store, "__minns_on_schedule__") {
                Ok(f) => f,
                Err(_) => return Ok(()),
            };

        self.usage.record_call();

        let life_before = store.get_fuel().unwrap_or(0);
        let result_code = schedule_fn
            .call(&mut store, schedule_id)
            .map_err(WasmError::from)?;

        let life_after = store.get_fuel().unwrap_or(0);
        self.usage
            .record_life(life_before.saturating_sub(life_after));

        if result_code < 0 {
            tracing::warn!(
                "schedule {} on module '{}' returned error code {}",
                schedule_id,
                self.descriptor.name,
                result_code
            );
        }

        Ok(())
    }

    /// Get the module name from the descriptor.
    pub fn name(&self) -> &str {
        &self.descriptor.name
    }
}
