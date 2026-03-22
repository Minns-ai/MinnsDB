//! MinnsDB WASM Module SDK
//!
//! Provides the types, host function bindings, proc macros, and ABI glue needed
//! to build agent modules that run inside MinnsDB's sandboxed WASM runtime.
//!
//! # Quick start (macro approach)
//!
//! ```rust,ignore
//! use minns_sdk::prelude::*;
//!
//! #[minns_module(name = "my_agent", version = "0.1.0")]
//! mod my_agent {
//!     use super::*;
//!
//!     #[minns_function]
//!     pub fn greet(name: String) -> String {
//!         format!("Hello, {}!", name)
//!     }
//!
//!     #[minns_function]
//!     pub fn count_nodes() -> Result<QueryResult, String> {
//!         graph_query_exec("MATCH (n) RETURN count(n) AS total")
//!     }
//! }
//! ```
//!
//! # Manual approach
//!
//! For full control, see the `abi` module to write raw `#[no_mangle]` exports.

pub mod abi;
pub mod host;
pub mod types;

/// Re-export proc macros so `use minns_sdk::prelude::*` brings them in scope.
pub use minns_sdk_macros::{minns_function, minns_module};

/// Re-exports for convenience.
pub mod prelude {
    pub use crate::abi::{
        descriptor_len, read_args, read_args_raw, register_descriptor, set_result, set_result_bytes,
    };
    pub use crate::host::{
        self, error, get_group_id, get_module_id, graph_query_exec, http_fetch_exec, info, log_msg,
        table_delete_row, table_get_row, table_insert_row, table_query_exec, warn,
    };
    pub use crate::types::*;
    pub use crate::{minns_function, minns_module};
}
