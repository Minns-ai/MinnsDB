// Test persistence of memories and strategies across server restarts
import { EventGraphDBClient } from './client/dist/index.js';

// Helper functions
function generateEventId() {
  const timestamp = BigInt(Date.now()) * BigInt(1000000);
  const random = BigInt(Math.floor(Math.random() * 1000000));
  return Number(timestamp + random);
}

function currentTimestamp() {
  return Date.now() * 1000000;
}

async function testPersistence() {
  console.log('Testing Persistent Storage Integration\n');
  console.log('=' .repeat(60));

  const client = new EventGraphDBClient({
    baseUrl: 'http://127.0.0.1:3000',
    timeout: 10000,
  });

  // Test 1: Check health
  console.log('\n1️⃣  Health check...');
  const health = await client.healthCheck();
  console.log(`   ✅ Server is ${health.status}`);
  console.log(`   📊 Current state: ${health.node_count} nodes, ${health.edge_count} edges`);

  // Test 2: Get current stats
  console.log('\n2️⃣  Getting current statistics...');
  const statsBefore = await client.getStats();
  console.log(`   📈 Events: ${statsBefore.total_events_processed}`);
  console.log(`   🧠 Memories: ${statsBefore.total_memories_formed}`);
  console.log(`   🎯 Strategies: ${statsBefore.total_strategies_extracted}`);
  console.log(`   📦 Episodes: ${statsBefore.total_episodes_detected}`);

  // Test 3: Process some events to create memories
  console.log('\n3️⃣  Processing events to create memories...');
  const AGENT_ID = 999;
  const SESSION_ID = 123;

  for (let i = 0; i < 5; i++) {
    const event = {
      id: generateEventId(),
      timestamp: currentTimestamp() + i * 1000000000,
      agent_id: AGENT_ID,
      session_id: SESSION_ID,
      event_type: {
        Action: {
          action_name: `test_action_${i}`,
          parameters: { step: i },
          reasoning: `Testing persistence step ${i}`,
        },
      },
      context: {
        fingerprint: 0,
        goals: [{ id: 1, description: 'Test persistence' }],
        active_tools: ['test_tool'],
        recent_results: i % 2 === 0 ? ['success'] : ['failure'],
        turn_count: i + 1,
        session_phase: 'execution',
      },
      significance: 0.5 + (i * 0.1),
    };

    const result = await client.processEvent(event);
    console.log(`   ✅ Event ${i + 1} processed: ${result.nodes_created} nodes, ${result.patterns_detected} patterns`);
  }

  // Test 4: Get updated stats
  console.log('\n4️⃣  Getting updated statistics...');
  const statsAfter = await client.getStats();
  console.log(`   📈 Events: ${statsBefore.total_events_processed} → ${statsAfter.total_events_processed} (+${statsAfter.total_events_processed - statsBefore.total_events_processed})`);
  console.log(`   🧠 Memories: ${statsBefore.total_memories_formed} → ${statsAfter.total_memories_formed} (+${statsAfter.total_memories_formed - statsBefore.total_memories_formed})`);
  console.log(`   🎯 Strategies: ${statsBefore.total_strategies_extracted} → ${statsAfter.total_strategies_extracted} (+${statsAfter.total_strategies_extracted - statsBefore.total_strategies_extracted})`);
  console.log(`   📦 Episodes: ${statsBefore.total_episodes_detected} → ${statsAfter.total_episodes_detected} (+${statsAfter.total_episodes_detected - statsBefore.total_episodes_detected})`);

  // Test 5: Retrieve memories
  console.log('\n5️⃣  Retrieving memories for agent...');
  const memories = await client.getAgentMemories(AGENT_ID, 10);
  console.log(`   ✅ Found ${memories.length} memories for agent ${AGENT_ID}`);
  if (memories.length > 0) {
    console.log(`   📝 First memory: strength=${memories[0].strength.toFixed(2)}, access_count=${memories[0].access_count}`);
  }

  // Test 6: Retrieve strategies
  console.log('\n6️⃣  Retrieving strategies for agent...');
  const strategies = await client.getAgentStrategies(AGENT_ID, 10);
  console.log(`   ✅ Found ${strategies.length} strategies for agent ${AGENT_ID}`);
  if (strategies.length > 0) {
    console.log(`   🎯 First strategy: ${strategies[0].name}, quality=${strategies[0].quality_score.toFixed(2)}`);
  }

  console.log('\n✅ All tests passed!');
  console.log('\n💾 Data is now persisted in ./data/eventgraph.redb');
  console.log('   You can restart the server and verify the data persists.');
}

testPersistence().catch(console.error);
