use minns_sdk::prelude::*;

#[minns_module(name = "hello-agent", version = "0.1.0")]
mod agent {
    use super::*;

    #[minns_function]
    pub fn greet(name: String) -> String {
        info(&format!("greet called for: {}", name));
        format!("Hello, {}! This is hello-agent running inside MinnsDB.", name)
    }

    #[minns_function]
    pub fn count_nodes() -> Result<QueryResult, String> {
        graph_query_exec("MATCH (n) RETURN count(n) AS total")
    }

    #[minns_function]
    pub fn insert_event(message: String) -> Result<InsertResult, String> {
        let values = vec!["hello-agent".to_string(), message];
        table_insert_row("events", &values)
    }

    #[minns_function]
    pub fn add(a: i64, b: i64) -> i64 {
        a + b
    }
}
