//! Backend-agnostic vector store. One trait ([`VectorStore`]) plus the value
//! types it operates on, with a single Qdrant-backed implementation
//! ([`QdrantStore`]).

mod error;
mod filter;
mod point;
mod qdrant;
mod query;
mod store;

pub use error::{VectorError, VectorResult};
pub use filter::Filter;
pub use point::{FilterValue, Payload, Point, ScoredPoint};
pub use qdrant::{
    BinaryEncoding, BinaryQuantization, BinaryQueryEncoding, Distance, ProductCompression,
    ProductQuantization, QdrantConfig, QdrantStore, Quantization, ScalarQuantization, TurboBits,
    TurboQuantization,
};
pub use query::{Query, QueryBuilder};
pub use store::VectorStore;
