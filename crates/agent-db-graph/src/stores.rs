//! Store boundaries for memory and strategy layers.
//!
//! These abstractions keep the graph engine independent from persistence.

use agent_db_core::types::{AgentId, ContextHash, SessionId};
use agent_db_events::core::EventContext;

use crate::episodes::Episode;
use crate::memory::{Memory, MemoryFormation, MemoryFormationConfig, MemoryId, MemoryStats, MemoryUpsert};
use crate::strategies::{
    Strategy, StrategyExtractor, StrategyExtractionConfig, StrategyId, StrategyStats,
    StrategySimilarityQuery, StrategyUpsert,
};

pub trait MemoryStore: Send + Sync {
    fn store_episode(&mut self, episode: &Episode) -> Option<MemoryUpsert>;
    fn get_memory(&self, memory_id: MemoryId) -> Option<Memory>;
    fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory>;
    fn retrieve_by_context(&mut self, context: &EventContext, limit: usize) -> Vec<Memory>;
    fn retrieve_by_context_similar(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
    ) -> Vec<Memory>;
    fn apply_outcome(&mut self, memory_id: MemoryId, success: bool) -> bool;
    fn get_stats(&self) -> MemoryStats;
    fn apply_decay(&mut self);
}

pub struct InMemoryMemoryStore {
    inner: MemoryFormation,
}

impl InMemoryMemoryStore {
    pub fn new(config: MemoryFormationConfig) -> Self {
        Self {
            inner: MemoryFormation::new(config),
        }
    }
}

impl MemoryStore for InMemoryMemoryStore {
    fn store_episode(&mut self, episode: &Episode) -> Option<MemoryUpsert> {
        self.inner.form_memory(episode)
    }

    fn get_memory(&self, memory_id: MemoryId) -> Option<Memory> {
        self.inner.get_memory(memory_id).cloned()
    }

    fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory> {
        self.inner.retrieve_by_agent(agent_id, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    fn retrieve_by_context(&mut self, context: &EventContext, limit: usize) -> Vec<Memory> {
        self.inner.retrieve_by_context(context, limit)
    }

    fn retrieve_by_context_similar(
        &mut self,
        context: &EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
    ) -> Vec<Memory> {
        self.inner
            .retrieve_by_context_similar(context, limit, min_similarity, agent_id, session_id)
    }

    fn apply_outcome(&mut self, memory_id: MemoryId, success: bool) -> bool {
        self.inner.apply_outcome(memory_id, success)
    }

    fn get_stats(&self) -> MemoryStats {
        self.inner.get_stats()
    }

    fn apply_decay(&mut self) {
        self.inner.apply_decay();
    }
}

pub trait StrategyStore: Send + Sync {
    fn store_episode(&mut self, episode: &Episode, events: &[agent_db_events::core::Event]) -> crate::GraphResult<Option<StrategyUpsert>>;
    fn get_strategy(&self, strategy_id: StrategyId) -> Option<Strategy>;
    fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy>;
    fn get_strategies_for_context(&self, context_hash: ContextHash, limit: usize) -> Vec<Strategy>;
    fn find_similar_strategies(&self, query: StrategySimilarityQuery) -> Vec<(Strategy, f32)>;
    fn update_strategy_outcome(&mut self, strategy_id: StrategyId, success: bool) -> crate::GraphResult<()>;
    fn get_stats(&self) -> StrategyStats;
}

pub struct InMemoryStrategyStore {
    inner: StrategyExtractor,
}

impl InMemoryStrategyStore {
    pub fn new(config: StrategyExtractionConfig) -> Self {
        Self {
            inner: StrategyExtractor::new(config),
        }
    }
}

impl StrategyStore for InMemoryStrategyStore {
    fn store_episode(
        &mut self,
        episode: &Episode,
        events: &[agent_db_events::core::Event],
    ) -> crate::GraphResult<Option<StrategyUpsert>> {
        self.inner.extract_from_episode(episode, events)
    }

    fn get_strategy(&self, strategy_id: StrategyId) -> Option<Strategy> {
        self.inner.get_strategy(strategy_id).cloned()
    }

    fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy> {
        self.inner
            .get_agent_strategies(agent_id, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    fn get_strategies_for_context(&self, context_hash: ContextHash, limit: usize) -> Vec<Strategy> {
        self.inner
            .get_strategies_for_context(context_hash, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    fn find_similar_strategies(&self, query: StrategySimilarityQuery) -> Vec<(Strategy, f32)> {
        self.inner.find_similar_strategies(query)
    }

    fn update_strategy_outcome(&mut self, strategy_id: StrategyId, success: bool) -> crate::GraphResult<()> {
        self.inner.update_strategy_outcome(strategy_id, success)
    }

    fn get_stats(&self) -> StrategyStats {
        self.inner.get_stats()
    }
}
