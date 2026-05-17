//! Qdrant-backed [`VectorStore`]. All translation between this crate's value
//! types and `qdrant_client`'s wire types is contained here.

use std::collections::HashMap;

use async_trait::async_trait;
use qdrant_client::qdrant::binary_quantization_query_encoding::Setting as QdrantBqQueryEncoding;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::quantization_config::Quantization as QdrantQuantization;
use qdrant_client::qdrant::value::Kind as ValueKind;
use qdrant_client::qdrant::vector_output::Vector as VectorOutputKind;
use qdrant_client::qdrant::{
    BinaryQuantization as QdrantBinaryQuant, BinaryQuantizationEncoding as QdrantBqEncoding,
    BinaryQuantizationQueryEncoding as QdrantBqQueryEncodingMsg, CompressionRatio, Condition,
    CountPointsBuilder, CreateCollectionBuilder, DeletePointsBuilder, Distance as QdrantDistance,
    Filter as QdrantFilter, GetCollectionInfoResponse, GetPointsBuilder, PointId, PointStruct,
    PointsIdsList, ProductQuantization as QdrantProductQuant, QueryPointsBuilder, RetrievedPoint,
    ScalarQuantization as QdrantScalarQuant, ScalarQuantizationBuilder, TurboQuantBitSize,
    TurboQuantization as QdrantTurboQuant, UpsertPointsBuilder, Value, VectorParamsBuilder,
    VectorsOutput,
};
use qdrant_client::Qdrant;

use crate::error::{VectorError, VectorResult};
use crate::filter::Filter;
use crate::point::{FilterValue, Payload, Point, ScoredPoint};
use crate::query::Query;
use crate::store::VectorStore;

/// Distance metric used to compare query and stored vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    /// Cosine similarity. Higher score is more similar.
    Cosine,
    /// Dot product. Higher score is more similar.
    Dot,
    /// Euclidean distance. Lower score is more similar.
    Euclid,
}

impl Distance {
    fn to_qdrant(self) -> QdrantDistance {
        match self {
            Distance::Cosine => QdrantDistance::Cosine,
            Distance::Dot => QdrantDistance::Dot,
            Distance::Euclid => QdrantDistance::Euclid,
        }
    }
}

/// Quantization applied to stored vectors. The choice is consequential: it
/// fixes the trade between RAM, recall, and search latency for the lifetime
/// of the collection.
///
/// At d = 1536 the per-vector size is roughly:
///
/// | Variant                       | Bytes | Compression |
/// |-------------------------------|-------|-------------|
/// | None (fp32)                   | 6144  | 1×          |
/// | [`Quantization::Scalar`] int8 | 1536  | 4×          |
/// | [`Quantization::Product`] X16 | 384   | 16×         |
/// | [`Quantization::Turbo`] 2-bit | 384   | 16×         |
/// | [`Quantization::Binary`] 1-bit| 192   | 32×         |
///
/// Pick the variant that satisfies the recall floor for the corpus you are
/// indexing; tune `always_ram` to pin the quantized vectors in memory for
/// fast scoring on the hot path.
#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    Scalar(ScalarQuantization),
    Product(ProductQuantization),
    Binary(BinaryQuantization),
    Turbo(TurboQuantization),
}

/// Scalar quantization parameters. Quantizes each fp32 component to int8 with
/// per-collection calibrated ranges. Roughly 4× compression with a small,
/// usually-imperceptible recall loss.
#[derive(Debug, Clone, Copy)]
pub struct ScalarQuantization {
    /// Quantile in `[0.5, 1.0]` used to clip outlier component values during
    /// calibration. `None` lets Qdrant pick its default (1.0). Lower values
    /// trade a tiny amount of accuracy for tighter quantization buckets.
    pub quantile: Option<f32>,
    /// Pin quantized vectors in RAM regardless of the main storage config.
    /// Strongly recommended on hot collections.
    pub always_ram: bool,
}

/// Compression ratio for [`Quantization::Product`]. Trains a per-shard
/// codebook; recall degrades as the ratio rises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductCompression {
    X4,
    X8,
    X16,
    X32,
    X64,
}

impl ProductCompression {
    fn to_qdrant(self) -> CompressionRatio {
        match self {
            ProductCompression::X4 => CompressionRatio::X4,
            ProductCompression::X8 => CompressionRatio::X8,
            ProductCompression::X16 => CompressionRatio::X16,
            ProductCompression::X32 => CompressionRatio::X32,
            ProductCompression::X64 => CompressionRatio::X64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProductQuantization {
    pub compression: ProductCompression,
    pub always_ram: bool,
}

/// How many bits per dimension binary quantization stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryEncoding {
    /// One bit per dimension (32× compression at fp32 input). Most aggressive.
    OneBit,
    /// One and a half bits per dimension.
    OneAndHalfBits,
    /// Two bits per dimension (16× compression).
    TwoBits,
}

impl BinaryEncoding {
    fn to_qdrant(self) -> QdrantBqEncoding {
        match self {
            BinaryEncoding::OneBit => QdrantBqEncoding::OneBit,
            BinaryEncoding::OneAndHalfBits => QdrantBqEncoding::OneAndHalfBits,
            BinaryEncoding::TwoBits => QdrantBqEncoding::TwoBits,
        }
    }
}

/// Asymmetric encoding for the query vector at search time. Pairing a finer
/// query encoding (e.g. `Scalar8Bits`) with a coarse stored encoding
/// (`OneBit`) recovers some of the recall lost to aggressive storage
/// quantization at the cost of slightly more compute per query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryQueryEncoding {
    /// Same encoding as the stored vector.
    Default,
    /// Binary-encode the query (fastest, lowest recall).
    Binary,
    /// 4-bit scalar query.
    Scalar4Bits,
    /// 8-bit scalar query (highest recall of the four).
    Scalar8Bits,
}

impl BinaryQueryEncoding {
    fn to_qdrant(self) -> QdrantBqQueryEncoding {
        match self {
            BinaryQueryEncoding::Default => QdrantBqQueryEncoding::Default,
            BinaryQueryEncoding::Binary => QdrantBqQueryEncoding::Binary,
            BinaryQueryEncoding::Scalar4Bits => QdrantBqQueryEncoding::Scalar4Bits,
            BinaryQueryEncoding::Scalar8Bits => QdrantBqQueryEncoding::Scalar8Bits,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryQuantization {
    pub encoding: BinaryEncoding,
    pub query_encoding: BinaryQueryEncoding,
    pub always_ram: bool,
}

/// Bits per dimension for [`Quantization::Turbo`]. TurboQuant is
/// data-oblivious (no codebook training) and matches the Shannon distortion
/// lower bound at 2-bit and 4-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboBits {
    Bits1,
    Bits1_5,
    Bits2,
    Bits4,
}

impl TurboBits {
    fn to_qdrant(self) -> TurboQuantBitSize {
        match self {
            TurboBits::Bits1 => TurboQuantBitSize::Bits1,
            TurboBits::Bits1_5 => TurboQuantBitSize::Bits15,
            TurboBits::Bits2 => TurboQuantBitSize::Bits2,
            TurboBits::Bits4 => TurboQuantBitSize::Bits4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TurboQuantization {
    pub bits: TurboBits,
    pub always_ram: bool,
}

impl Quantization {
    fn into_qdrant(self) -> QdrantQuantization {
        match self {
            Quantization::Scalar(s) => {
                let mut b = ScalarQuantizationBuilder::default().always_ram(s.always_ram);
                if let Some(q) = s.quantile {
                    b = b.quantile(q);
                }
                let q: QdrantScalarQuant = b.into();
                QdrantQuantization::Scalar(q)
            },
            Quantization::Product(p) => QdrantQuantization::Product(QdrantProductQuant {
                compression: p.compression.to_qdrant() as i32,
                always_ram: Some(p.always_ram),
            }),
            Quantization::Binary(b) => QdrantQuantization::Binary(QdrantBinaryQuant {
                always_ram: Some(b.always_ram),
                encoding: Some(b.encoding.to_qdrant() as i32),
                query_encoding: Some(QdrantBqQueryEncodingMsg {
                    variant: Some(
                        qdrant_client::qdrant::binary_quantization_query_encoding::Variant::Setting(
                            b.query_encoding.to_qdrant() as i32,
                        ),
                    ),
                }),
            }),
            Quantization::Turbo(t) => QdrantQuantization::Turboquant(QdrantTurboQuant {
                always_ram: Some(t.always_ram),
                bits: Some(t.bits.to_qdrant() as i32),
            }),
        }
    }
}

/// Configuration for opening a single Qdrant collection.
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection: String,
    pub dim: usize,
    pub distance: Distance,
    pub quantization: Option<Quantization>,
}

impl QdrantConfig {
    /// Construct with cosine distance and no quantization. Production
    /// callers should pick a quantization explicitly via
    /// [`Self::with_quantization`].
    pub fn new(url: impl Into<String>, collection: impl Into<String>, dim: usize) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            collection: collection.into(),
            dim,
            distance: Distance::Cosine,
            quantization: None,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = Some(quantization);
        self
    }
}

/// Qdrant-backed [`VectorStore`].
pub struct QdrantStore {
    client: Qdrant,
    collection: String,
    dim: usize,
}

impl QdrantStore {
    /// Open or create the configured collection. Existing collections are
    /// reused unchanged; we never silently re-configure a live collection.
    /// Verifies an existing collection's dim matches `config.dim` and
    /// returns [`VectorError::DimensionMismatch`] otherwise so a deploy with
    /// the wrong embedding model fails fast instead of corrupting search.
    pub async fn open(config: QdrantConfig) -> VectorResult<Self> {
        let mut builder = Qdrant::from_url(&config.url);
        if let Some(key) = config.api_key.as_deref() {
            builder = builder.api_key(key.to_owned());
        }
        let client = builder.build().map_err(backend_error)?;

        let exists = client
            .collection_exists(&config.collection)
            .await
            .map_err(backend_error)?;

        if !exists {
            tracing::info!(
                collection = %config.collection,
                dim = config.dim,
                "creating Qdrant collection"
            );
            let mut create =
                CreateCollectionBuilder::new(config.collection.clone()).vectors_config(
                    VectorParamsBuilder::new(config.dim as u64, config.distance.to_qdrant()),
                );
            if let Some(q) = config.quantization {
                create = create.quantization_config(q.into_qdrant());
            }
            client
                .create_collection(create)
                .await
                .map_err(backend_error)?;
        } else {
            let info = client
                .collection_info(&config.collection)
                .await
                .map_err(backend_error)?;
            let actual_dim = extract_collection_dim(&info)?;
            if actual_dim != config.dim {
                return Err(VectorError::DimensionMismatch {
                    expected: config.dim,
                    got: actual_dim,
                });
            }
        }

        Ok(Self {
            client,
            collection: config.collection,
            dim: config.dim,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn collection_name(&self) -> &str {
        &self.collection
    }

    fn check_dim(&self, vector: &[f32]) -> VectorResult<()> {
        if vector.len() != self.dim {
            return Err(VectorError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn upsert(&self, points: Vec<Point>) -> VectorResult<()> {
        if points.is_empty() {
            return Ok(());
        }
        for p in &points {
            self.check_dim(&p.vector)?;
        }
        let qpoints: Vec<PointStruct> = points.into_iter().map(point_to_qdrant).collect();
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection, qpoints).wait(true))
            .await
            .map_err(backend_error)?;
        Ok(())
    }

    async fn fetch(&self, ids: &[u128]) -> VectorResult<Vec<Option<Point>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let qids: Vec<PointId> = ids.iter().copied().map(id_to_point_id).collect();
        let resp = self
            .client
            .get_points(
                GetPointsBuilder::new(&self.collection, qids)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(backend_error)?;

        // Qdrant returns only the ids it finds, in arbitrary order. Index by
        // id, then map back to the input order so the result is positionally
        // aligned with `ids`.
        let mut by_id: HashMap<u128, Point> = HashMap::with_capacity(resp.result.len());
        for retrieved in resp.result {
            let id = retrieved
                .id
                .as_ref()
                .ok_or_else(|| VectorError::Backend("retrieved point missing id".into()))
                .and_then(point_id_to_id)?;
            let vector = vector_from_retrieved(&retrieved)?;
            let payload = payload_from_qdrant(retrieved.payload);
            by_id.insert(id, Point::new(id, vector, payload));
        }
        Ok(ids.iter().map(|id| by_id.remove(id)).collect())
    }

    async fn delete(&self, ids: &[u128]) -> VectorResult<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let qids: Vec<PointId> = ids.iter().copied().map(id_to_point_id).collect();
        let selector = PointsSelectorOneOf::Points(PointsIdsList { ids: qids });
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(selector)
                    .wait(true),
            )
            .await
            .map_err(backend_error)?;
        // Qdrant's delete response does not include a precise removed count.
        // Returning `ids.len()` is a non-binding upper bound documented on
        // the trait method; callers treat it as a diagnostic hint.
        Ok(ids.len())
    }

    async fn search(&self, query: &Query) -> VectorResult<Vec<ScoredPoint>> {
        self.check_dim(query.vector())?;
        if query.top_k() == 0 {
            return Err(VectorError::InvalidQuery("top_k must be >= 1".into()));
        }
        let mut builder = QueryPointsBuilder::new(&self.collection)
            .query(query.vector().to_vec())
            .limit(query.top_k() as u64)
            .with_payload(true);
        if let Some(min) = query.min_score() {
            builder = builder.score_threshold(min);
        }
        if let Some(f) = query.filter() {
            if !f.is_empty() {
                builder = builder.filter(filter_to_qdrant(f));
            }
        }
        let resp = self.client.query(builder).await.map_err(backend_error)?;
        let mut out = Vec::with_capacity(resp.result.len());
        for sp in resp.result {
            let id = match sp.id.as_ref() {
                Some(pid) => point_id_to_id(pid)?,
                None => continue,
            };
            out.push(ScoredPoint {
                id,
                score: sp.score,
                payload: payload_from_qdrant(sp.payload),
            });
        }
        Ok(out)
    }

    async fn count(&self) -> VectorResult<usize> {
        let resp = self
            .client
            .count(CountPointsBuilder::new(&self.collection).exact(true))
            .await
            .map_err(backend_error)?;
        Ok(resp.result.map(|r| r.count as usize).unwrap_or(0))
    }
}

fn backend_error<E: std::fmt::Display>(e: E) -> VectorError {
    VectorError::Backend(e.to_string())
}

/// Encode a 128-bit id as a UUID string for Qdrant's wire format. The
/// canonical 16-byte UUID layout round-trips exactly.
fn id_to_point_id(id: u128) -> PointId {
    let uuid = uuid::Uuid::from_bytes(id.to_be_bytes());
    PointId::from(uuid.to_string())
}

fn point_id_to_id(pid: &PointId) -> VectorResult<u128> {
    match pid.point_id_options.as_ref() {
        Some(PointIdOptions::Uuid(s)) => {
            let uuid = uuid::Uuid::parse_str(s)
                .map_err(|e| VectorError::Backend(format!("invalid UUID point id: {e}")))?;
            Ok(u128::from_be_bytes(*uuid.as_bytes()))
        },
        // Accept numeric ids so a collection written by something else is
        // not catastrophic to open.
        Some(PointIdOptions::Num(n)) => Ok(u128::from(*n)),
        None => Err(VectorError::Backend("point id missing".into())),
    }
}

fn point_to_qdrant(p: Point) -> PointStruct {
    PointStruct::new(
        id_to_point_id(p.id),
        p.vector,
        payload_to_qdrant(&p.payload),
    )
}

fn filter_value_to_qdrant(v: &FilterValue) -> Value {
    let kind = match v {
        // Qdrant payloads use signed integers; a u64 above i64::MAX would
        // wrap. Callers that need values that large should pack them as a
        // string and accept the loss of integer comparison semantics.
        FilterValue::U64(n) => ValueKind::IntegerValue(*n as i64),
        FilterValue::I64(n) => ValueKind::IntegerValue(*n),
        FilterValue::Bool(b) => ValueKind::BoolValue(*b),
        FilterValue::Str(s) => ValueKind::StringValue(s.clone()),
    };
    Value { kind: Some(kind) }
}

fn payload_to_qdrant(payload: &Payload) -> HashMap<String, Value> {
    payload
        .iter()
        .map(|(k, v)| (k.to_owned(), filter_value_to_qdrant(v)))
        .collect()
}

fn payload_from_qdrant(map: HashMap<String, Value>) -> Payload {
    let mut payload = Payload::new();
    for (k, v) in map {
        if let Some(value) = filter_value_from_qdrant(&v) {
            payload.insert(k, value);
        }
    }
    payload
}

fn filter_value_from_qdrant(v: &Value) -> Option<FilterValue> {
    match v.kind.as_ref()? {
        ValueKind::IntegerValue(n) => Some(FilterValue::I64(*n)),
        ValueKind::BoolValue(b) => Some(FilterValue::Bool(*b)),
        ValueKind::StringValue(s) => Some(FilterValue::Str(s.clone())),
        // Doubles, lists, structs, nulls fall outside the crate's payload
        // schema. They occur only if the collection was written by another
        // client and we treat them as absent rather than guess a coercion.
        _ => None,
    }
}

fn filter_to_qdrant(f: &Filter) -> QdrantFilter {
    QdrantFilter {
        must: f
            .eq_conditions()
            .iter()
            .map(|(field, val)| eq_condition(field, val))
            .collect(),
        must_not: f
            .neq_conditions()
            .iter()
            .map(|(field, val)| eq_condition(field, val))
            .collect(),
        ..Default::default()
    }
}

fn eq_condition(field: &str, value: &FilterValue) -> Condition {
    match value {
        FilterValue::U64(n) => Condition::matches(field, *n as i64),
        FilterValue::I64(n) => Condition::matches(field, *n),
        FilterValue::Bool(b) => Condition::matches(field, *b),
        FilterValue::Str(s) => Condition::matches(field, s.clone()),
    }
}

fn vector_from_retrieved(rp: &RetrievedPoint) -> VectorResult<Vec<f32>> {
    let vector = rp
        .vectors
        .as_ref()
        .and_then(VectorsOutput::get_vector)
        .ok_or_else(|| {
            VectorError::Backend("retrieved point has no vector (with_vectors not honored)".into())
        })?;
    match vector {
        VectorOutputKind::Dense(d) => Ok(d.data),
        VectorOutputKind::Sparse(_) | VectorOutputKind::MultiDense(_) => Err(VectorError::Backend(
            "only dense vectors are supported".into(),
        )),
    }
}

fn extract_collection_dim(info: &GetCollectionInfoResponse) -> VectorResult<usize> {
    use qdrant_client::qdrant::vectors_config::Config as VectorsConfigKind;

    let collection = info
        .result
        .as_ref()
        .ok_or_else(|| VectorError::Backend("collection info missing result".into()))?;
    let config = collection
        .config
        .as_ref()
        .ok_or_else(|| VectorError::Backend("collection info missing config".into()))?;
    let params = config
        .params
        .as_ref()
        .ok_or_else(|| VectorError::Backend("collection params missing".into()))?;
    let vectors_config = params
        .vectors_config
        .as_ref()
        .ok_or_else(|| VectorError::Backend("collection vectors_config missing".into()))?;
    let kind = vectors_config
        .config
        .as_ref()
        .ok_or_else(|| VectorError::Backend("collection vectors_config kind missing".into()))?;
    match kind {
        VectorsConfigKind::Params(p) => Ok(p.size as usize),
        VectorsConfigKind::ParamsMap(_) => Err(VectorError::Backend(
            "named vectors are not supported by this crate".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_round_trips_through_uuid_for_endpoints_and_random_values() {
        for id in [
            0u128,
            1,
            u128::MAX,
            0x0123_4567_89ab_cdef_0123_4567_89ab_cdef,
            42,
        ] {
            let pid = id_to_point_id(id);
            let back = point_id_to_id(&pid).expect("decodes");
            assert_eq!(back, id, "round trip failed for {id:#x}");
        }
    }

    #[test]
    fn numeric_point_id_decodes_as_u128() {
        let pid = PointId {
            point_id_options: Some(PointIdOptions::Num(7)),
        };
        assert_eq!(point_id_to_id(&pid).unwrap(), 7);
    }

    #[test]
    fn missing_point_id_is_a_backend_error() {
        let pid = PointId {
            point_id_options: None,
        };
        assert!(matches!(point_id_to_id(&pid), Err(VectorError::Backend(_))));
    }

    #[test]
    fn filter_value_round_trips_for_supported_kinds() {
        for v in [
            FilterValue::U64(123),
            FilterValue::I64(-456),
            FilterValue::Bool(true),
            FilterValue::Str("hello".into()),
        ] {
            let qv = filter_value_to_qdrant(&v);
            let back = filter_value_from_qdrant(&qv).expect("decodes");
            let expected = match v {
                FilterValue::U64(n) => FilterValue::I64(n as i64),
                other => other,
            };
            assert_eq!(back, expected);
        }
    }

    #[test]
    fn empty_payload_round_trips() {
        let empty = payload_to_qdrant(&Payload::EMPTY);
        assert!(empty.is_empty());
        assert!(payload_from_qdrant(empty).is_empty());
    }

    #[test]
    fn payload_round_trips_through_qdrant_value_map() {
        let payload = Payload::new()
            .with("agent_id", 7u64)
            .with("tier", "episodic")
            .with("archived", false);
        let qmap = payload_to_qdrant(&payload);
        assert_eq!(qmap.len(), 3);
        let back = payload_from_qdrant(qmap);
        assert_eq!(back.get("agent_id"), Some(&FilterValue::I64(7)));
        assert_eq!(back.get("tier"), Some(&FilterValue::Str("episodic".into())));
        assert_eq!(back.get("archived"), Some(&FilterValue::Bool(false)));
    }

    #[test]
    fn empty_filter_translation_has_no_conditions() {
        let q = filter_to_qdrant(&Filter::new());
        assert!(q.must.is_empty());
        assert!(q.must_not.is_empty());
    }

    #[test]
    fn filter_translation_separates_eq_and_neq() {
        let f = Filter::new()
            .eq("agent_id", 5u64)
            .eq("tier", "semantic")
            .neq("status", "archived");
        let q = filter_to_qdrant(&f);
        assert_eq!(q.must.len(), 2);
        assert_eq!(q.must_not.len(), 1);
    }

    #[test]
    fn scalar_quantization_threads_quantile_when_set() {
        let q = Quantization::Scalar(ScalarQuantization {
            quantile: Some(0.99),
            always_ram: true,
        });
        match q.into_qdrant() {
            QdrantQuantization::Scalar(s) => {
                assert_eq!(s.quantile, Some(0.99));
                assert_eq!(s.always_ram, Some(true));
            },
            other => panic!("expected scalar quantization, got {other:?}"),
        }
    }

    #[test]
    fn product_quantization_carries_compression() {
        let q = Quantization::Product(ProductQuantization {
            compression: ProductCompression::X16,
            always_ram: true,
        });
        match q.into_qdrant() {
            QdrantQuantization::Product(p) => {
                assert_eq!(p.compression, CompressionRatio::X16 as i32);
                assert_eq!(p.always_ram, Some(true));
            },
            other => panic!("expected product quantization, got {other:?}"),
        }
    }

    #[test]
    fn binary_quantization_carries_encoding_and_query_encoding() {
        let q = Quantization::Binary(BinaryQuantization {
            encoding: BinaryEncoding::TwoBits,
            query_encoding: BinaryQueryEncoding::Scalar8Bits,
            always_ram: true,
        });
        match q.into_qdrant() {
            QdrantQuantization::Binary(b) => {
                assert_eq!(b.encoding, Some(QdrantBqEncoding::TwoBits as i32));
                assert_eq!(b.always_ram, Some(true));
                assert!(b.query_encoding.is_some());
            },
            other => panic!("expected binary quantization, got {other:?}"),
        }
    }

    #[test]
    fn turbo_quantization_carries_bits_2() {
        let q = Quantization::Turbo(TurboQuantization {
            bits: TurboBits::Bits2,
            always_ram: true,
        });
        match q.into_qdrant() {
            QdrantQuantization::Turboquant(t) => {
                assert_eq!(t.bits, Some(TurboQuantBitSize::Bits2 as i32));
                assert_eq!(t.always_ram, Some(true));
            },
            other => panic!("expected turbo quantization, got {other:?}"),
        }
    }

    #[test]
    fn product_compression_enum_maps_to_qdrant_variants() {
        for (ours, theirs) in [
            (ProductCompression::X4, CompressionRatio::X4),
            (ProductCompression::X8, CompressionRatio::X8),
            (ProductCompression::X16, CompressionRatio::X16),
            (ProductCompression::X32, CompressionRatio::X32),
            (ProductCompression::X64, CompressionRatio::X64),
        ] {
            assert_eq!(ours.to_qdrant() as i32, theirs as i32);
        }
    }

    #[test]
    fn turbo_bits_enum_maps_to_qdrant_variants() {
        for (ours, theirs) in [
            (TurboBits::Bits1, TurboQuantBitSize::Bits1),
            (TurboBits::Bits1_5, TurboQuantBitSize::Bits15),
            (TurboBits::Bits2, TurboQuantBitSize::Bits2),
            (TurboBits::Bits4, TurboQuantBitSize::Bits4),
        ] {
            assert_eq!(ours.to_qdrant() as i32, theirs as i32);
        }
    }
}
