// EventGraphDB Integration Test
// Tests the complete stack: Rust server + JS client

const { EventGraphDBClient } = require('./client/dist/index');

// Helper to generate unique IDs
// EventId in Rust is u128, so we use a large number
function generateEventId() {
  // Use timestamp in nanoseconds + random component
  // This creates a unique u128-compatible number
  const timestamp = BigInt(Date.now()) * BigInt(1000000); // Convert to nanoseconds
  const random = BigInt(Math.floor(Math.random() * 1000000));
  return Number(timestamp + random);
}

// Helper to get current timestamp in nanoseconds
function currentTimestamp() {
  return Date.now() * 1000000; // Convert ms to ns
}

// Helper to hash a string to a u64 number
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

// Test runner
async function runTests() {
  console.log('🧪 EventGraphDB Integration Test Suite\n');
  console.log('=' .repeat(60));

  const client = new EventGraphDBClient({
    baseUrl: 'http://127.0.0.1:3000',
    timeout: 10000,
  });

  let passed = 0;
  let failed = 0;

  // Test 1: Health Check
  console.log('\n1️⃣  Testing health check endpoint...');
  try {
    const health = await client.healthCheck();
    if (health.is_healthy && health.version) {
      console.log(`   ✅ Server healthy: v${health.version}`);
      console.log(`   📊 Graph size: ${health.node_count} nodes, ${health.edge_count} edges`);
      passed++;
    } else {
      console.log('   ❌ Server unhealthy');
      failed++;
    }
  } catch (error) {
    console.log(`   ❌ Health check failed: ${error.message}`);
    failed++;
    return; // Can't continue if server is down
  }

  // Test 2: Process Event (Code Debugging Session)
  console.log('\n2️⃣  Testing event processing...');
  try {
    const event = {
      id: generateEventId(),
      timestamp: currentTimestamp(),
      agent_id: 1,
      agent_type: 'code-debugger',
      session_id: 42,
      event_type: {
        Action: {
          action_name: 'add_null_check',
          parameters: {
            variable: 'user',
            location: 'getUserById()',
          },
          outcome: {
            Success: {
              result: { tests_pass: true, null_errors: 0 },
            },
          },
          duration_ns: 2000000000,
        },
      },
      causality_chain: [],
      context: {
        environment: {
          variables: {
            language: 'javascript',
            framework: 'node',
            test_framework: 'jest',
            error_type: 'null_reference',
            file: 'user-service.js',
          },
          spatial: null,
          temporal: {
            time_of_day: null,
            deadlines: [],
            patterns: [],
          },
        },
        active_goals: [
          {
            id: 1,
            description: 'Fix null reference error in user service',
            priority: 0.9,
            deadline: null,
            progress: 0.5,
            subgoals: [],
          },
        ],
        resources: {
          computational: {
            cpu_percent: 45.0,
            memory_bytes: 8589934592,
            storage_bytes: 107374182400,
            network_bandwidth: 1000000000,
          },
          external: {},
        },
        // fingerprint auto-computed by server
        embeddings: null,
      },
      metadata: {
        severity: { String: 'high' },
        confidence: { String: '0.9' },
      },
    };

    // Debug: Log the event being sent
    console.log(`   🔍 Debug: Sending event with ID: ${event.id}`);

    const result = await client.processEvent(event);
    if (result.success) {
      console.log(`   ✅ Event processed successfully`);
      console.log(`   📦 Created ${result.nodes_created} nodes`);
      console.log(`   🔍 Detected ${result.patterns_detected} patterns`);
      console.log(`   ⏱️  Processing time: ${result.processing_time_ms}ms`);
      passed++;
    } else {
      console.log('   ❌ Event processing failed');
      failed++;
    }
  } catch (error) {
    console.log(`   ❌ Event processing error: ${error.message}`);
    console.log(`   🔍 Debug: Error type: ${error.constructor.name}`);
    if (error.statusCode) {
      console.log(`   🔍 Debug: Status code: ${error.statusCode}`);
    }
    if (error.details) {
      console.log(`   🔍 Debug: Details: ${error.details}`);
    }
    console.log(`   🔍 Debug: Full error:`, error);
    failed++;
  }

  // Test 3: Process Multiple Events (to build up data)
  console.log('\n3️⃣  Processing multiple events to build graph...');
  try {
    const events = [
      {
        id: generateEventId(),
        timestamp: currentTimestamp(),
        agent_id: 1,
        agent_type: 'code-debugger',
        session_id: 42,
        event_type: {
          Action: {
            action_name: 'add_try_catch',
            parameters: { location: 'asyncFetch()' },
            outcome: { Success: { result: { errors_caught: 5 } } },
            duration_ns: 1500000000,
          },
        },
        causality_chain: [],
        context: {
          environment: {
            variables: {
              language: 'javascript',
              error_type: 'unhandled_promise',
              async_function: 'asyncFetch',
            },
            spatial: null,
            temporal: {
              time_of_day: null,
              deadlines: [],
              patterns: [],
            },
          },
          active_goals: [
            {
              id: 2,
              description: 'Handle async errors properly',
              priority: 0.8,
              deadline: null,
              progress: 0.3,
              subgoals: [],
            },
          ],
          resources: {
            computational: {
              cpu_percent: 35.0,
              memory_bytes: 8589934592,
              storage_bytes: 107374182400,
              network_bandwidth: 1000000000,
            },
            external: {},
          },
          // fingerprint auto-computed by server
          embeddings: null,
        },
        metadata: {},
      },
      {
        id: generateEventId(),
        timestamp: currentTimestamp(),
        agent_id: 1,
        agent_type: 'code-debugger',
        session_id: 43,
        event_type: {
          Cognitive: {
            process_type: 'Reasoning',
            input: { problem: 'performance bottleneck' },
            output: { solution: 'add caching' },
            reasoning_trace: [
              'Identified repeated database queries',
              'Considered caching strategies',
              'Selected Redis for session cache',
            ],
          },
        },
        causality_chain: [],
        context: {
          environment: {
            variables: {
              language: 'javascript',
              database: 'postgresql',
              issue_type: 'performance',
              bottleneck: 'repeated_queries',
            },
            spatial: null,
            temporal: {
              time_of_day: null,
              deadlines: [],
              patterns: [],
            },
          },
          active_goals: [
            {
              id: 3,
              description: 'Optimize database query performance',
              priority: 0.7,
              deadline: null,
              progress: 0.6,
              subgoals: [],
            },
          ],
          resources: {
            computational: {
              cpu_percent: 65.0,
              memory_bytes: 8589934592,
              storage_bytes: 107374182400,
              network_bandwidth: 1000000000,
            },
            external: {},
          },
          // fingerprint auto-computed by server
          embeddings: null,
        },
        metadata: {},
      },
    ];

    let eventsProcessed = 0;
    for (const event of events) {
      const result = await client.processEvent(event);
      if (result.success) {
        eventsProcessed++;
      }
    }

    if (eventsProcessed === events.length) {
      console.log(`   ✅ Processed ${eventsProcessed} events successfully`);
      passed++;
    } else {
      console.log(`   ⚠️  Processed ${eventsProcessed}/${events.length} events`);
      failed++;
    }
  } catch (error) {
    console.log(`   ❌ Multiple event processing error: ${error.message}`);
    failed++;
  }

  // Test 4: Query Agent Memories
  console.log('\n4️⃣  Testing memory retrieval...');
  try {
    const memories = await client.getAgentMemories(1, 10);
    console.log(`   ✅ Retrieved ${memories.length} memories`);
    if (memories.length > 0) {
      const memory = memories[0];
      console.log(`   📝 Memory ${memory.id}:`);
      console.log(`      - Strength: ${memory.strength.toFixed(2)}`);
      console.log(`      - Relevance: ${memory.relevance_score.toFixed(2)}`);
      console.log(`      - Accessed: ${memory.access_count} times`);
    }
    passed++;
  } catch (error) {
    console.log(`   ❌ Memory retrieval error: ${error.message}`);
    failed++;
  }

  // Test 5: Query Agent Strategies
  console.log('\n5️⃣  Testing strategy retrieval...');
  try {
    const strategies = await client.getAgentStrategies(1, 10);
    console.log(`   ✅ Retrieved ${strategies.length} strategies`);
    if (strategies.length > 0) {
      const strategy = strategies[0];
      const successRate = (strategy.success_count / (strategy.success_count + strategy.failure_count) * 100).toFixed(1);
      console.log(`   📋 Strategy: "${strategy.name}"`);
      console.log(`      - Quality: ${strategy.quality_score.toFixed(2)}`);
      console.log(`      - Success rate: ${successRate}%`);
      console.log(`      - Steps: ${strategy.reasoning_steps.length}`);
    }
    passed++;
  } catch (error) {
    console.log(`   ❌ Strategy retrieval error: ${error.message}`);
    failed++;
  }

  // Test 6: Get Episodes
  console.log('\n6️⃣  Testing episode retrieval...');
  try {
    const episodes = await client.getEpisodes(10);
    console.log(`   ✅ Retrieved ${episodes.length} episodes`);
    if (episodes.length > 0) {
      const episode = episodes[0];
      console.log(`   📺 Episode ${episode.id}:`);
      console.log(`      - Events: ${episode.event_count}`);
      console.log(`      - Significance: ${episode.significance.toFixed(2)}`);
      if (episode.outcome) {
        console.log(`      - Outcome: ${episode.outcome}`);
      }
    }
    passed++;
  } catch (error) {
    console.log(`   ❌ Episode retrieval error: ${error.message}`);
    failed++;
  }

  // Test 7: Get System Statistics
  console.log('\n7️⃣  Testing system statistics...');
  try {
    const stats = await client.getStats();
    console.log(`   ✅ System statistics retrieved`);
    console.log(`   📊 Total events processed: ${stats.total_events_processed}`);
    console.log(`   🧠 Memories formed: ${stats.total_memories_formed}`);
    console.log(`   📋 Strategies extracted: ${stats.total_strategies_extracted}`);
    console.log(`   🔁 Reinforcements applied: ${stats.total_reinforcements_applied}`);
    console.log(`   📺 Episodes detected: ${stats.total_episodes_detected}`);
    console.log(`   ⏱️  Avg processing time: ${stats.average_processing_time_ms.toFixed(2)}ms`);
    passed++;
  } catch (error) {
    console.log(`   ❌ Statistics retrieval error: ${error.message}`);
    failed++;
  }

  // Test 8: Get Action Suggestions (Policy Guide)
  console.log('\n8️⃣  Testing action suggestions (Policy Guide)...');
  try {
    // Use a context hash that might exist from our events
    const suggestions = await client.getActionSuggestions(hashString('null_reference_error_javascript'), undefined, 5);
    console.log(`   ✅ Retrieved ${suggestions.length} action suggestions`);
    if (suggestions.length > 0) {
      console.log(`   💡 Top suggestions for "null_reference_error":`);
      suggestions.slice(0, 3).forEach((s, i) => {
        console.log(`      ${i + 1}. ${s.action_name} (${(s.success_probability * 100).toFixed(1)}% success)`);
        console.log(`         Evidence: ${s.evidence_count} cases`);
      });
    } else {
      console.log('   ℹ️  No suggestions yet (needs more data)');
    }
    passed++;
  } catch (error) {
    console.log(`   ❌ Action suggestions error: ${error.message}`);
    failed++;
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('\n📊 Test Results:');
  console.log(`   ✅ Passed: ${passed}`);
  console.log(`   ❌ Failed: ${failed}`);
  console.log(`   📈 Success Rate: ${((passed / (passed + failed)) * 100).toFixed(1)}%`);

  if (failed === 0) {
    console.log('\n🎉 All tests passed! EventGraphDB is working correctly!\n');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests failed. Please check the errors above.\n');
    process.exit(1);
  }
}

// Run tests
console.log('\n🚀 Starting EventGraphDB Integration Tests...');
console.log('⏳ Connecting to server at http://127.0.0.1:3000...\n');

// Give user a moment to ensure server is running
setTimeout(runTests, 1000);
