//! Unified diff parser

use crate::error::AstResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDiff {
    pub files: Vec<DiffFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffFile {
    pub old_path: Option<String>,
    pub new_path: Option<String>,
    pub status: DiffFileStatus,
    pub hunks: Vec<DiffHunk>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffFileStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    pub old_start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    pub lines: Vec<DiffLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub content: String,
    pub old_line_number: Option<usize>,
    pub new_line_number: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffLineKind {
    Added,
    Removed,
    Context,
}

/// Parse a unified diff text into structured form.
pub fn parse_unified_diff(diff_text: &str) -> AstResult<ParsedDiff> {
    let mut files = Vec::new();
    let mut current_file: Option<DiffFile> = None;
    let mut current_hunk: Option<DiffHunk> = None;
    let mut old_line: usize = 0;
    let mut new_line: usize = 0;

    for line in diff_text.lines() {
        if line.starts_with("diff --git") || line.starts_with("diff -") {
            // Flush current hunk and file
            if let Some(hunk) = current_hunk.take() {
                if let Some(ref mut file) = current_file {
                    file.hunks.push(hunk);
                }
            }
            if let Some(file) = current_file.take() {
                files.push(file);
            }
            current_file = Some(DiffFile {
                old_path: None,
                new_path: None,
                status: DiffFileStatus::Modified,
                hunks: Vec::new(),
            });
        } else if line.starts_with("--- ") {
            if let Some(ref mut file) = current_file {
                let path = line.strip_prefix("--- ").unwrap_or("").trim();
                if path == "/dev/null" {
                    file.old_path = None;
                    file.status = DiffFileStatus::Added;
                } else {
                    file.old_path = Some(path.strip_prefix("a/").unwrap_or(path).to_string());
                }
            }
        } else if line.starts_with("+++ ") {
            if let Some(ref mut file) = current_file {
                let path = line.strip_prefix("+++ ").unwrap_or("").trim();
                if path == "/dev/null" {
                    file.new_path = None;
                    file.status = DiffFileStatus::Deleted;
                } else {
                    file.new_path = Some(path.strip_prefix("b/").unwrap_or(path).to_string());
                }
            }
        } else if line.starts_with("@@ ") {
            // Flush current hunk
            if let Some(hunk) = current_hunk.take() {
                if let Some(ref mut file) = current_file {
                    file.hunks.push(hunk);
                }
            }

            // Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            if let Some((old_start, old_count, new_start, new_count)) = parse_hunk_header(line) {
                old_line = old_start;
                new_line = new_start;
                current_hunk = Some(DiffHunk {
                    old_start,
                    old_count,
                    new_start,
                    new_count,
                    lines: Vec::new(),
                });
            }
        } else if let Some(ref mut hunk) = current_hunk {
            if let Some(content) = line.strip_prefix('+') {
                hunk.lines.push(DiffLine {
                    kind: DiffLineKind::Added,
                    content: content.to_string(),
                    old_line_number: None,
                    new_line_number: Some(new_line),
                });
                new_line += 1;
            } else if let Some(content) = line.strip_prefix('-') {
                hunk.lines.push(DiffLine {
                    kind: DiffLineKind::Removed,
                    content: content.to_string(),
                    old_line_number: Some(old_line),
                    new_line_number: None,
                });
                old_line += 1;
            } else if let Some(content) = line.strip_prefix(' ') {
                hunk.lines.push(DiffLine {
                    kind: DiffLineKind::Context,
                    content: content.to_string(),
                    old_line_number: Some(old_line),
                    new_line_number: Some(new_line),
                });
                old_line += 1;
                new_line += 1;
            } else if line == "\\ No newline at end of file" {
                // Skip this line
            } else {
                // Context line without space prefix (some diff formats)
                hunk.lines.push(DiffLine {
                    kind: DiffLineKind::Context,
                    content: line.to_string(),
                    old_line_number: Some(old_line),
                    new_line_number: Some(new_line),
                });
                old_line += 1;
                new_line += 1;
            }
        }
    }

    // Flush remaining
    if let Some(hunk) = current_hunk.take() {
        if let Some(ref mut file) = current_file {
            file.hunks.push(hunk);
        }
    }
    if let Some(file) = current_file.take() {
        files.push(file);
    }

    // Detect renames (old_path != new_path and both are Some)
    for file in &mut files {
        if let (Some(old), Some(new)) = (&file.old_path, &file.new_path) {
            if old != new && file.status == DiffFileStatus::Modified {
                file.status = DiffFileStatus::Renamed;
            }
        }
    }

    Ok(ParsedDiff { files })
}

fn parse_hunk_header(line: &str) -> Option<(usize, usize, usize, usize)> {
    // @@ -old_start,old_count +new_start,new_count @@
    let stripped = line.strip_prefix("@@ ")?;
    let end = stripped.find(" @@")?;
    let ranges = &stripped[..end];

    let mut parts = ranges.split(' ');
    let old_part = parts.next()?.strip_prefix('-')?;
    let new_part = parts.next()?.strip_prefix('+')?;

    let (old_start, old_count) = parse_range(old_part)?;
    let (new_start, new_count) = parse_range(new_part)?;

    Some((old_start, old_count, new_start, new_count))
}

fn parse_range(range: &str) -> Option<(usize, usize)> {
    if let Some((start, count)) = range.split_once(',') {
        Some((start.parse().ok()?, count.parse().ok()?))
    } else {
        // Single line: "N" means start=N, count=1
        Some((range.parse().ok()?, 1))
    }
}
