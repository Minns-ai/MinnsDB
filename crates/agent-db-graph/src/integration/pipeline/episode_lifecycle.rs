// crates/agent-db-graph/src/integration/pipeline/episode_lifecycle.rs
//
// Reinforcement learning, world model training, and transition model updates.

use super::*;

impl GraphEngine {
    /// Process episode for reinforcement learning
    pub(crate) async fn process_episode_for_reinforcement(
        &self,
        episode: &Episode,
    ) -> GraphResult<()> {
        // Determine success/failure
        let success = matches!(episode.outcome, Some(EpisodeOutcome::Success));

        // Calculate duration from events
        let duration_seconds = {
            let store = self.event_store.read().await;
            if let (Some(start_event), Some(end_event_id)) =
                (store.get(&episode.start_event), episode.end_event)
            {
                if let Some(end_event) = store.get(&end_event_id) {
                    let duration_ns = end_event.timestamp.saturating_sub(start_event.timestamp);
                    (duration_ns as f32) / 1_000_000_000.0
                } else {
                    1.0 // Default
                }
            } else {
                1.0 // Default
            }
        };

        // Calculate metrics
        let metrics = EpisodeMetrics {
            duration_seconds,
            expected_duration_seconds: 5.0, // Default expectation
            quality_score: Some(episode.significance),
            custom_metrics: HashMap::new(),
        };

        // Apply reinforcement
        {
            let mut inference = self.inference.write().await;
            let _result = inference
                .reinforce_patterns(episode, success, Some(metrics))
                .await?;
        }
        // Drop inference write lock before calling update_transition_model
        // which acquires its own locks (event_store, transition_model)
        self.update_transition_model(episode).await?;

        self.stats
            .total_reinforcements_applied
            .fetch_add(1, AtomicOrdering::Relaxed);

        Ok(())
    }

    /// Process episode for world model training (shadow mode).
    /// Assembles a training tuple and submits it to the critic.
    pub(crate) async fn process_episode_for_world_model(
        &self,
        episode: &Episode,
    ) -> GraphResult<()> {
        let Some(ref wm) = self.world_model else {
            return Ok(());
        };

        // Load events for feature extraction
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|id| store.get(id).cloned())
                .collect()
        };

        // Find best matching memory for this episode's context
        let memory = {
            let mut store = self.memory_store.write().await;
            store
                .retrieve_by_context(&episode.context, 1)
                .into_iter()
                .next()
        };

        // Find matching strategy by context hash
        let strategy = {
            let store = self.strategy_store.read().await;
            store
                .get_strategies_for_context(episode.context_signature, 1)
                .into_iter()
                .next()
        };

        // Assemble training tuple
        if let Some(tuple) = world_model::assemble_training_tuple(
            episode,
            &events,
            memory.as_ref(),
            strategy.as_ref(),
        ) {
            let mut wm_guard = wm.write().await;
            wm_guard.submit_training(tuple);
            tracing::debug!(
                "World model training tuple submitted episode_id={} events={} salience={:.3}",
                episode.id,
                events.len(),
                episode.salience_score,
            );
        }

        Ok(())
    }

    pub(crate) async fn update_transition_model(&self, episode: &Episode) -> GraphResult<()> {
        let should_update = matches!(
            episode.outcome,
            Some(EpisodeOutcome::Success) | Some(EpisodeOutcome::Failure)
        );
        if !should_update {
            return Ok(());
        }

        let success = matches!(episode.outcome, Some(EpisodeOutcome::Success));
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|event_id| store.get(event_id).cloned())
                .collect()
        };

        if events.is_empty() {
            return Ok(());
        }

        let outputs = crate::contracts::build_learning_outputs(episode, &events);
        let mut model = self.transition_model.write().await;
        model.update_from_trace(
            outputs.episode_record.goal_bucket_id,
            &outputs.abstract_trace,
            outputs.episode_record.episode_id,
            success,
        );
        tracing::info!(
            "Transition model updated episode_id={} goal_bucket_id={} transitions={} success={}",
            outputs.episode_record.episode_id,
            outputs.episode_record.goal_bucket_id,
            outputs.abstract_trace.transitions.len(),
            success
        );

        Ok(())
    }
}
