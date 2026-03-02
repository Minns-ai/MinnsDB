//! Fixture-style unit tests for AST parsing and diff parsing

#[cfg(test)]
mod rust_parsing {
    use crate::entities::*;
    use crate::parser::AstParser;

    fn parse_rust(source: &str) -> crate::parser::ParseResult {
        let parser = AstParser::new();
        parser.parse(source, "rust", "test.rs").unwrap()
    }

    #[test]
    fn test_free_function() {
        let source = r#"
fn add(x: i32, y: i32) -> i32 {
    x + y
}
"#;
        let result = parse_rust(source);
        assert_eq!(result.entities.len(), 1);

        let func = &result.entities[0];
        assert_eq!(func.name, "add");
        assert_eq!(func.kind, CodeEntityKind::Function);
        assert_eq!(func.language, "rust");
        assert!(func.qualified_name.contains("add"));
        assert_eq!(func.parameters.len(), 2);
        assert_eq!(func.parameters[0].name, "x");
        assert_eq!(func.parameters[0].type_annotation.as_deref(), Some("i32"));
        assert_eq!(func.parameters[1].name, "y");
        assert_eq!(func.return_type.as_deref(), Some("i32"));
    }

    #[test]
    fn test_struct_with_fields() {
        let source = r#"
pub struct Point {
    pub x: f64,
    pub y: f64,
}
"#;
        let result = parse_rust(source);

        // Should have: Struct + 2 Fields
        let structs: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Struct)
            .collect();
        assert_eq!(structs.len(), 1);
        assert_eq!(structs[0].name, "Point");
        assert_eq!(structs[0].visibility.as_deref(), Some("pub"));

        let fields: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Field)
            .collect();
        assert_eq!(fields.len(), 2);

        // Should have FieldOf relationships
        let field_of_rels: Vec<_> = result
            .relationships
            .iter()
            .filter(|r| r.kind == CodeRelationKind::FieldOf)
            .collect();
        assert_eq!(field_of_rels.len(), 2);
        assert_eq!(field_of_rels[0].confidence, 1.0);
    }

    #[test]
    fn test_impl_with_methods() {
        let source = r#"
struct Foo;

impl Foo {
    fn bar(&self) -> String {
        String::new()
    }

    fn baz(&self, x: i32) {}
}
"#;
        let result = parse_rust(source);

        let methods: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Method)
            .collect();
        assert_eq!(methods.len(), 2);
        assert_eq!(methods[0].name, "bar");
        assert_eq!(methods[1].name, "baz");

        // Should have Contains relationships (Foo contains bar, baz)
        let contains_rels: Vec<_> = result
            .relationships
            .iter()
            .filter(|r| r.kind == CodeRelationKind::Contains)
            .collect();
        assert!(
            contains_rels.len() >= 2,
            "expected at least 2 Contains relationships, got {}",
            contains_rels.len()
        );
    }

    #[test]
    fn test_use_imports() {
        let source = r#"
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
"#;
        let result = parse_rust(source);

        let imports: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Import)
            .collect();
        assert!(
            !imports.is_empty(),
            "expected at least one Import entity, got none"
        );
    }

    #[test]
    fn test_trait_definition() {
        let source = r#"
pub trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
}
"#;
        let result = parse_rust(source);

        let traits: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Interface)
            .collect();
        assert_eq!(traits.len(), 1);
        assert_eq!(traits[0].name, "Drawable");
        assert_eq!(traits[0].visibility.as_deref(), Some("pub"));
    }

    #[test]
    fn test_enum_definition() {
        let source = r#"
pub enum Color {
    Red,
    Green,
    Blue,
}
"#;
        let result = parse_rust(source);

        let enums: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Enum)
            .collect();
        assert_eq!(enums.len(), 1);
        assert_eq!(enums[0].name, "Color");

        let variants: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::EnumVariant)
            .collect();
        assert_eq!(variants.len(), 3);
    }

    #[test]
    fn test_doc_comment_extraction() {
        let source = r#"
/// This is a documented function.
/// It does important things.
pub fn documented() -> bool {
    true
}
"#;
        let result = parse_rust(source);
        assert_eq!(result.entities.len(), 1);
        let func = &result.entities[0];
        assert!(
            func.doc_comment.is_some(),
            "expected doc comment to be extracted"
        );
        let doc = func.doc_comment.as_ref().unwrap();
        assert!(doc.contains("documented function"));
    }

    #[test]
    fn test_const_and_static() {
        let source = r#"
const MAX_SIZE: usize = 100;
static GLOBAL_COUNT: u32 = 0;
"#;
        let result = parse_rust(source);

        let consts: Vec<_> = result
            .entities
            .iter()
            .filter(|e| e.kind == CodeEntityKind::Constant)
            .collect();
        assert!(!consts.is_empty(), "expected at least one Constant entity");
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(
            AstParser::detect_language("foo.rs"),
            Some("rust".to_string())
        );
        assert_eq!(
            AstParser::detect_language("bar.py"),
            Some("python".to_string())
        );
        assert_eq!(
            AstParser::detect_language("baz.js"),
            Some("javascript".to_string())
        );
        assert_eq!(
            AstParser::detect_language("qux.ts"),
            Some("typescript".to_string())
        );
        assert_eq!(
            AstParser::detect_language("main.go"),
            Some("go".to_string())
        );
        assert_eq!(AstParser::detect_language("readme.md"), None);
        assert_eq!(AstParser::detect_language("data.json"), None);
    }

    #[test]
    fn test_impl_trait_relationship() {
        let source = r#"
trait Animal {
    fn speak(&self);
}

struct Dog;

impl Animal for Dog {
    fn speak(&self) {}
}
"#;
        let result = parse_rust(source);

        let implements_rels: Vec<_> = result
            .relationships
            .iter()
            .filter(|r| r.kind == CodeRelationKind::Implements)
            .collect();
        assert!(
            !implements_rels.is_empty(),
            "expected at least one Implements relationship"
        );
        // Heuristic relationship should have confidence < 1.0
        for rel in &implements_rels {
            assert!(
                rel.confidence < 1.0,
                "Implements should be heuristic (confidence < 1.0), got {}",
                rel.confidence
            );
        }
    }
}

#[cfg(test)]
mod diff_parsing {
    use crate::diff::*;

    #[test]
    fn test_added_file() {
        let diff = r#"diff --git a/new_file.rs b/new_file.rs
--- /dev/null
+++ b/new_file.rs
@@ -0,0 +1,3 @@
+fn hello() {
+    println!("hello");
+}
"#;
        let result = parse_unified_diff(diff).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].status, DiffFileStatus::Added);
        assert_eq!(result.files[0].new_path.as_deref(), Some("new_file.rs"));
        assert_eq!(result.files[0].hunks.len(), 1);
        assert_eq!(result.files[0].hunks[0].lines.len(), 3);
        assert!(result.files[0].hunks[0]
            .lines
            .iter()
            .all(|l| l.kind == DiffLineKind::Added));
    }

    #[test]
    fn test_deleted_file() {
        let diff = r#"diff --git a/old_file.rs b/old_file.rs
--- a/old_file.rs
+++ /dev/null
@@ -1,2 +0,0 @@
-fn old() {}
-fn deprecated() {}
"#;
        let result = parse_unified_diff(diff).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].status, DiffFileStatus::Deleted);
        assert_eq!(result.files[0].old_path.as_deref(), Some("old_file.rs"));
        assert_eq!(result.files[0].hunks[0].lines.len(), 2);
        assert!(result.files[0].hunks[0]
            .lines
            .iter()
            .all(|l| l.kind == DiffLineKind::Removed));
    }

    #[test]
    fn test_modified_file_with_context() {
        let diff = r#"diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,5 +1,5 @@
 fn main() {
-    println!("old");
+    println!("new");
     // context
 }
"#;
        let result = parse_unified_diff(diff).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].status, DiffFileStatus::Modified);

        let hunk = &result.files[0].hunks[0];
        let added: Vec<_> = hunk
            .lines
            .iter()
            .filter(|l| l.kind == DiffLineKind::Added)
            .collect();
        let removed: Vec<_> = hunk
            .lines
            .iter()
            .filter(|l| l.kind == DiffLineKind::Removed)
            .collect();
        let context: Vec<_> = hunk
            .lines
            .iter()
            .filter(|l| l.kind == DiffLineKind::Context)
            .collect();

        assert_eq!(added.len(), 1);
        assert_eq!(removed.len(), 1);
        assert!(context.len() >= 2);
    }

    #[test]
    fn test_multiple_files() {
        let diff = r#"diff --git a/file1.rs b/file1.rs
--- a/file1.rs
+++ b/file1.rs
@@ -1,3 +1,3 @@
-fn old1() {}
+fn new1() {}
diff --git a/file2.rs b/file2.rs
--- /dev/null
+++ b/file2.rs
@@ -0,0 +1,1 @@
+fn added() {}
"#;
        let result = parse_unified_diff(diff).unwrap();
        assert_eq!(result.files.len(), 2);
        assert_eq!(result.files[0].status, DiffFileStatus::Modified);
        assert_eq!(result.files[1].status, DiffFileStatus::Added);
    }

    #[test]
    fn test_multiple_hunks() {
        let diff = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,3 @@
-fn first_old() {}
+fn first_new() {}
@@ -10,3 +10,3 @@
-fn second_old() {}
+fn second_new() {}
"#;
        let result = parse_unified_diff(diff).unwrap();
        assert_eq!(result.files.len(), 1);
        assert_eq!(result.files[0].hunks.len(), 2);
    }
}
