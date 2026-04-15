use super::token::{Spanned, Token};

pub struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

#[derive(Debug)]
pub struct LexError {
    pub message: String,
    pub position: usize,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lex error at byte {}: {}", self.position, self.message)
    }
}

impl std::error::Error for LexError {}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    pub fn tokenize(input: &str) -> Result<Vec<Spanned>, LexError> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let spanned = lexer.next_token()?;
            let is_eof = spanned.token == Token::Eof;
            tokens.push(spanned);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Spanned, LexError> {
        self.skip_whitespace_and_comments();

        if self.at_end() {
            return Ok(Spanned {
                token: Token::Eof,
                span: (self.pos, self.pos),
            });
        }

        let ch = self.peek().unwrap();

        // Identifiers and keywords
        if ch.is_ascii_alphabetic() || ch == b'_' {
            return Ok(self.scan_ident_or_keyword());
        }

        // Numbers
        if ch.is_ascii_digit() {
            return self.scan_number();
        }

        // String literals
        if ch == b'"' {
            return self.scan_string();
        }

        // Backtick-quoted identifiers: `column name` or `where` (escape hatch
        // for genuinely reserved keywords like WHERE, RETURN, SELECT that
        // would otherwise conflict with query syntax). The content between
        // backticks is taken verbatim and emitted as Token::Ident.
        if ch == b'`' {
            return self.scan_quoted_ident();
        }

        // Multi-character symbols
        let start = self.pos;

        match ch {
            b'(' => {
                self.advance();
                Ok(Spanned {
                    token: Token::LParen,
                    span: (start, self.pos),
                })
            },
            b')' => {
                self.advance();
                Ok(Spanned {
                    token: Token::RParen,
                    span: (start, self.pos),
                })
            },
            b'{' => {
                self.advance();
                Ok(Spanned {
                    token: Token::LBrace,
                    span: (start, self.pos),
                })
            },
            b'}' => {
                self.advance();
                Ok(Spanned {
                    token: Token::RBrace,
                    span: (start, self.pos),
                })
            },
            b'[' => {
                self.advance();
                Ok(Spanned {
                    token: Token::LBracket,
                    span: (start, self.pos),
                })
            },
            b']' => {
                self.advance();
                Ok(Spanned {
                    token: Token::RBracket,
                    span: (start, self.pos),
                })
            },
            b':' => {
                self.advance();
                Ok(Spanned {
                    token: Token::Colon,
                    span: (start, self.pos),
                })
            },
            b',' => {
                self.advance();
                Ok(Spanned {
                    token: Token::Comma,
                    span: (start, self.pos),
                })
            },
            b'*' => {
                self.advance();
                Ok(Spanned {
                    token: Token::Star,
                    span: (start, self.pos),
                })
            },
            b'.' => {
                self.advance();
                if self.peek() == Some(b'.') {
                    self.advance();
                    Ok(Spanned {
                        token: Token::DotDot,
                        span: (start, self.pos),
                    })
                } else {
                    Ok(Spanned {
                        token: Token::Dot,
                        span: (start, self.pos),
                    })
                }
            },
            b'-' => {
                self.advance();
                if self.peek() == Some(b'>') {
                    self.advance();
                    Ok(Spanned {
                        token: Token::Arrow,
                        span: (start, self.pos),
                    })
                } else {
                    Ok(Spanned {
                        token: Token::Dash,
                        span: (start, self.pos),
                    })
                }
            },
            b'<' => {
                self.advance();
                if self.peek() == Some(b'-') {
                    self.advance();
                    Ok(Spanned {
                        token: Token::BackArrow,
                        span: (start, self.pos),
                    })
                } else if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(Spanned {
                        token: Token::Lte,
                        span: (start, self.pos),
                    })
                } else {
                    Ok(Spanned {
                        token: Token::Lt,
                        span: (start, self.pos),
                    })
                }
            },
            b'>' => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(Spanned {
                        token: Token::Gte,
                        span: (start, self.pos),
                    })
                } else {
                    Ok(Spanned {
                        token: Token::Gt,
                        span: (start, self.pos),
                    })
                }
            },
            b'=' => {
                self.advance();
                Ok(Spanned {
                    token: Token::Eq,
                    span: (start, self.pos),
                })
            },
            b'!' => {
                self.advance();
                if self.peek() == Some(b'=') {
                    self.advance();
                    Ok(Spanned {
                        token: Token::Neq,
                        span: (start, self.pos),
                    })
                } else {
                    Err(LexError {
                        message: "expected '=' after '!'".into(),
                        position: start,
                    })
                }
            },
            _ => Err(LexError {
                message: format!("unexpected character '{}'", ch as char),
                position: start,
            }),
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while let Some(ch) = self.peek() {
                if ch.is_ascii_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }

            // Skip line comments
            if self.pos + 1 < self.input.len()
                && self.input[self.pos] == b'/'
                && self.input[self.pos + 1] == b'/'
            {
                while let Some(ch) = self.peek() {
                    self.advance();
                    if ch == b'\n' {
                        break;
                    }
                }
                continue;
            }

            break;
        }
    }

    fn scan_ident_or_keyword(&mut self) -> Spanned {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == b'_' {
                self.advance();
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
        let upper = text.to_ascii_uppercase();

        // Handle STARTS WITH as a two-word keyword
        if upper == "STARTS" {
            let saved_pos = self.pos;
            // Peek ahead past whitespace for WITH
            let mut peek_pos = self.pos;
            while peek_pos < self.input.len() && self.input[peek_pos].is_ascii_whitespace() {
                peek_pos += 1;
            }
            // Check if the next word is WITH
            let remaining = &self.input[peek_pos..];
            if remaining.len() >= 4 {
                let candidate = std::str::from_utf8(&remaining[..4]).unwrap_or("");
                if candidate.eq_ignore_ascii_case("WITH")
                    && (remaining.len() == 4
                        || !remaining[4].is_ascii_alphanumeric() && remaining[4] != b'_')
                {
                    // Consume through "WITH"
                    self.pos = peek_pos + 4;
                    return Spanned {
                        token: Token::StartsWith,
                        span: (start, self.pos),
                    };
                }
            }
            // Not followed by WITH, restore and emit as ident
            self.pos = saved_pos;
        }

        // Reserved keyword table.
        //
        // Only tokens the parser must pattern-match at fixed positions are
        // specialized: clause delimiters, operators, literal keywords, and
        // statement-top-level verbs. Everything else falls through to
        // `Token::Ident(text)` — including DDL-structural keywords like
        // `TABLE`, `PRIMARY`, `KEY`, `VALUES`, `SET`, etc. — and is recognized
        // by case-insensitive string comparison at the specific parser sites
        // that need it. The effect: a user column literally named `key` or
        // `table` parses as a normal identifier in every expression context
        // without any special handling.
        let token = match upper.as_str() {
            // Clause delimiters
            "MATCH" => Token::Match,
            "WHEN" => Token::When,
            "WHERE" => Token::Where,
            "RETURN" => Token::Return,
            "ORDER" => Token::Order,
            "BY" => Token::By,
            "LIMIT" => Token::Limit,
            "AS" => Token::As,
            "OF" => Token::Of,
            "DISTINCT" => Token::Distinct,
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "FROM" => Token::From,
            "JOIN" => Token::Join,
            "LEFT" => Token::Left,
            "INNER" => Token::Inner,
            "ON" => Token::On,
            "GROUP" => Token::Group,
            "HAVING" => Token::Having,
            // Expression operators
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "IS" => Token::Is,
            "IN" => Token::In,
            "BETWEEN" => Token::Between,
            "LIKE" => Token::Like,
            "CONTAINS" => Token::Contains,
            // Literal keywords
            "NULL" => Token::Null,
            "TRUE" => Token::True,
            "FALSE" => Token::False,
            // Statement-top-level verbs
            "CREATE" => Token::Create,
            "DROP" => Token::Drop,
            "INSERT" => Token::Insert,
            "UPDATE" => Token::Update,
            "DELETE" => Token::Delete,
            "ALTER" => Token::Alter,
            "SUBSCRIBE" => Token::Subscribe,
            "UNSUBSCRIBE" => Token::Unsubscribe,
            // Everything else is a soft keyword / identifier
            _ => Token::Ident(text.to_string()),
        };

        Spanned {
            token,
            span: (start, self.pos),
        }
    }

    fn scan_number(&mut self) -> Result<Spanned, LexError> {
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        // Check for decimal point (but not `..`)
        if self.peek() == Some(b'.')
            && self.pos + 1 < self.input.len()
            && self.input[self.pos + 1] != b'.'
        {
            self.advance(); // consume '.'
            let has_frac = self.peek().is_some_and(|c| c.is_ascii_digit());
            if !has_frac {
                return Err(LexError {
                    message: "expected digit after decimal point".into(),
                    position: self.pos,
                });
            }
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
            let text = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
            let val: f64 = text.parse().map_err(|_| LexError {
                message: format!("invalid float literal '{}'", text),
                position: start,
            })?;
            return Ok(Spanned {
                token: Token::FloatLit(val),
                span: (start, self.pos),
            });
        }

        let text = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
        let val: i64 = text.parse().map_err(|_| LexError {
            message: format!("invalid integer literal '{}'", text),
            position: start,
        })?;
        Ok(Spanned {
            token: Token::IntLit(val),
            span: (start, self.pos),
        })
    }

    /// Scan a backtick-quoted identifier: `` `column name` ``.
    ///
    /// The content between backticks is taken verbatim and emitted as
    /// `Token::Ident(String)`. The quoting is the escape hatch for the
    /// narrow set of genuinely reserved keywords (WHERE, RETURN, FROM,
    /// AND, OR, NOT, IS, IN, BETWEEN, LIKE, CONTAINS, TRUE, FALSE, NULL,
    /// and the statement verbs) that could not be used as bare identifiers
    /// without changing the parse of another query. Soft keywords like
    /// `KEY`, `TABLE`, `PRIMARY` do not need backticks — they already
    /// lex as regular identifiers. A literal backtick inside the
    /// identifier is written `` `` `` (doubled).
    ///
    /// Unicode is preserved: the scanner accumulates raw bytes and decodes
    /// them as UTF-8 at the end, so `` `café` `` and `` `名前` `` round-trip
    /// correctly. The lexer input is already `&str` so the UTF-8 validation
    /// always succeeds, but we use the fallible decode defensively.
    fn scan_quoted_ident(&mut self) -> Result<Spanned, LexError> {
        let start = self.pos;
        self.advance(); // consume opening '`'
        let mut bytes: Vec<u8> = Vec::new();

        loop {
            if self.at_end() {
                return Err(LexError {
                    message: "unterminated quoted identifier".into(),
                    position: start,
                });
            }
            let ch = self.advance();
            if ch == b'`' {
                // Doubled backtick is a literal backtick in the identifier.
                if self.peek() == Some(b'`') {
                    self.advance();
                    bytes.push(b'`');
                    continue;
                }
                if bytes.is_empty() {
                    return Err(LexError {
                        message: "empty quoted identifier".into(),
                        position: start,
                    });
                }
                let value = String::from_utf8(bytes).map_err(|e| LexError {
                    message: format!("invalid UTF-8 in quoted identifier: {}", e),
                    position: start,
                })?;
                return Ok(Spanned {
                    token: Token::Ident(value),
                    span: (start, self.pos),
                });
            }
            bytes.push(ch);
        }
    }

    fn scan_string(&mut self) -> Result<Spanned, LexError> {
        let start = self.pos;
        self.advance(); // consume opening '"'
        let mut value = String::new();

        loop {
            if self.at_end() {
                return Err(LexError {
                    message: "unterminated string literal".into(),
                    position: start,
                });
            }
            let ch = self.advance();
            match ch {
                b'"' => {
                    return Ok(Spanned {
                        token: Token::StringLit(value),
                        span: (start, self.pos),
                    });
                },
                b'\\' => {
                    if self.at_end() {
                        return Err(LexError {
                            message: "unterminated escape in string".into(),
                            position: self.pos,
                        });
                    }
                    let esc = self.advance();
                    match esc {
                        b'"' => value.push('"'),
                        b'\\' => value.push('\\'),
                        b'n' => value.push('\n'),
                        b't' => value.push('\t'),
                        b'r' => value.push('\r'),
                        _ => {
                            value.push('\\');
                            value.push(esc as char);
                        },
                    }
                },
                _ => value.push(ch as char),
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        if self.pos < self.input.len() {
            Some(self.input[self.pos])
        } else {
            None
        }
    }

    fn advance(&mut self) -> u8 {
        let ch = self.input[self.pos];
        self.pos += 1;
        ch
    }

    fn at_end(&self) -> bool {
        self.pos >= self.input.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(input: &str) -> Vec<Token> {
        Lexer::tokenize(input)
            .unwrap()
            .into_iter()
            .map(|s| s.token)
            .collect()
    }

    #[test]
    fn test_keywords() {
        let tokens = tok("MATCH WHERE RETURN LIMIT");
        assert_eq!(
            tokens,
            vec![
                Token::Match,
                Token::Where,
                Token::Return,
                Token::Limit,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_keywords_case_insensitive() {
        let tokens = tok("match Where RETURN limit");
        assert_eq!(
            tokens,
            vec![
                Token::Match,
                Token::Where,
                Token::Return,
                Token::Limit,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_starts_with_keyword() {
        let tokens = tok("STARTS WITH");
        assert_eq!(tokens, vec![Token::StartsWith, Token::Eof]);
    }

    #[test]
    fn test_starts_without_with() {
        let tokens = tok("starts something");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("starts".into()),
                Token::Ident("something".into()),
                Token::Eof,
            ]
        );
    }

    // ── Soft-keyword regression tests ──────────────────────────────────
    //
    // These keywords used to be specialized Token variants. They now lex as
    // `Token::Ident`, which lets them act as user-chosen column, table, and
    // alias names throughout the query language without any special handling
    // in the parser. The lexer preserves the user's original casing.

    #[test]
    fn test_soft_keyword_key_lexes_as_ident() {
        let tokens = tok("key KEY Key");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("key".into()),
                Token::Ident("KEY".into()),
                Token::Ident("Key".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_soft_keyword_full_matrix() {
        // Every keyword converted from specialized variant to Ident in the
        // soft-keyword refactor. This test is the canonical list — adding
        // a new soft keyword means adding it here.
        let kws = [
            "TABLE",
            "PRIMARY",
            "KEY",
            "UNIQUE",
            "REFERENCES",
            "GRAPH",
            "DEFAULT",
            "AUTOINCREMENT",
            "INDEX",
            "COLUMN",
            "ADD",
            "INTO",
            "VALUES",
            "SET",
            "ALL",
            "LAST",
            "TO",
        ];
        for kw in kws {
            let tokens = tok(kw);
            assert_eq!(
                tokens,
                vec![Token::Ident(kw.to_string()), Token::Eof],
                "{} should lex as Ident",
                kw
            );
        }
    }

    #[test]
    fn test_reserved_keywords_still_specialized() {
        // Operators, clause delimiters, literal keywords, and statement
        // verbs must remain specialized — their Token::X variants are the
        // ones the parser pattern-matches at fixed positions.
        assert_eq!(tok("AND"), vec![Token::And, Token::Eof]);
        assert_eq!(tok("OR"), vec![Token::Or, Token::Eof]);
        assert_eq!(tok("NOT"), vec![Token::Not, Token::Eof]);
        assert_eq!(tok("IS"), vec![Token::Is, Token::Eof]);
        assert_eq!(tok("IN"), vec![Token::In, Token::Eof]);
        assert_eq!(tok("BETWEEN"), vec![Token::Between, Token::Eof]);
        assert_eq!(tok("LIKE"), vec![Token::Like, Token::Eof]);
        assert_eq!(tok("CONTAINS"), vec![Token::Contains, Token::Eof]);
        assert_eq!(tok("TRUE"), vec![Token::True, Token::Eof]);
        assert_eq!(tok("FALSE"), vec![Token::False, Token::Eof]);
        assert_eq!(tok("NULL"), vec![Token::Null, Token::Eof]);
        assert_eq!(tok("MATCH"), vec![Token::Match, Token::Eof]);
        assert_eq!(tok("WHERE"), vec![Token::Where, Token::Eof]);
        assert_eq!(tok("RETURN"), vec![Token::Return, Token::Eof]);
        assert_eq!(tok("FROM"), vec![Token::From, Token::Eof]);
        assert_eq!(tok("CREATE"), vec![Token::Create, Token::Eof]);
        assert_eq!(tok("INSERT"), vec![Token::Insert, Token::Eof]);
    }

    // ── Backtick-quoted identifier tests ────────────────────────────────

    #[test]
    fn test_backtick_quoted_ident_basic() {
        let tokens = tok("`column name`");
        assert_eq!(tokens, vec![Token::Ident("column name".into()), Token::Eof]);
    }

    #[test]
    fn test_backtick_quoted_reserved_keyword() {
        // The whole point of backtick quoting: WHERE is a reserved keyword
        // that would otherwise be impossible to use as an identifier.
        let tokens = tok("`where`");
        assert_eq!(tokens, vec![Token::Ident("where".into()), Token::Eof]);
    }

    #[test]
    fn test_backtick_quoted_preserves_case_and_spaces() {
        let tokens = tok("`My Column`");
        assert_eq!(tokens, vec![Token::Ident("My Column".into()), Token::Eof]);
    }

    #[test]
    fn test_backtick_quoted_doubled_backtick_is_literal() {
        let tokens = tok("`a``b`");
        assert_eq!(tokens, vec![Token::Ident("a`b".into()), Token::Eof]);
    }

    #[test]
    fn test_backtick_quoted_empty_is_error() {
        assert!(Lexer::tokenize("``").is_err());
    }

    #[test]
    fn test_backtick_quoted_unterminated_is_error() {
        assert!(Lexer::tokenize("`unterminated").is_err());
    }

    /// Unicode regression: backtick-quoted identifiers must preserve
    /// non-ASCII bytes exactly. Before the fix, `scan_quoted_ident` pushed
    /// bytes one-at-a-time as `u8 as char`, which corrupts multi-byte UTF-8
    /// sequences. After the fix, bytes are accumulated and decoded as
    /// UTF-8 at the end, so `café` and `名前` round-trip correctly.
    #[test]
    fn test_backtick_quoted_unicode_identifier() {
        let tokens = tok("`café`");
        assert_eq!(tokens, vec![Token::Ident("café".into()), Token::Eof]);

        let tokens = tok("`名前`");
        assert_eq!(tokens, vec![Token::Ident("名前".into()), Token::Eof]);

        // Mixed ASCII and non-ASCII with a doubled backtick.
        let tokens = tok("`a``é`");
        assert_eq!(tokens, vec![Token::Ident("a`é".into()), Token::Eof]);
    }

    #[test]
    fn test_string_literal() {
        let tokens = tok(r#""hello world""#);
        assert_eq!(
            tokens,
            vec![Token::StringLit("hello world".into()), Token::Eof]
        );
    }

    #[test]
    fn test_string_escape() {
        let tokens = tok(r#""say \"hi\"""#);
        assert_eq!(
            tokens,
            vec![Token::StringLit("say \"hi\"".into()), Token::Eof]
        );
    }

    #[test]
    fn test_integer_literal() {
        let tokens = tok("42 0 999");
        assert_eq!(
            tokens,
            vec![
                Token::IntLit(42),
                Token::IntLit(0),
                Token::IntLit(999),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_float_literal() {
        let tokens = tok("3.14 0.5");
        assert_eq!(
            tokens,
            vec![Token::FloatLit(3.14), Token::FloatLit(0.5), Token::Eof]
        );
    }

    #[test]
    fn test_arrow_and_comparison_operators() {
        let tokens = tok("-> <- < > <= >= = !=");
        assert_eq!(
            tokens,
            vec![
                Token::Arrow,
                Token::BackArrow,
                Token::Lt,
                Token::Gt,
                Token::Lte,
                Token::Gte,
                Token::Eq,
                Token::Neq,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_dot_dot_vs_dots() {
        let tokens = tok("a..b");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("a".into()),
                Token::DotDot,
                Token::Ident("b".into()),
                Token::Eof,
            ]
        );

        let tokens = tok("a.b");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("a".into()),
                Token::Dot,
                Token::Ident("b".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_full_query() {
        let tokens =
            tok(r#"MATCH (n:Person)-[r:KNOWS]->(m) WHERE n.name = "Alice" RETURN m.name LIMIT 10"#);
        assert_eq!(
            tokens,
            vec![
                Token::Match,
                Token::LParen,
                Token::Ident("n".into()),
                Token::Colon,
                Token::Ident("Person".into()),
                Token::RParen,
                Token::Dash,
                Token::LBracket,
                Token::Ident("r".into()),
                Token::Colon,
                Token::Ident("KNOWS".into()),
                Token::RBracket,
                Token::Arrow,
                Token::LParen,
                Token::Ident("m".into()),
                Token::RParen,
                Token::Where,
                Token::Ident("n".into()),
                Token::Dot,
                Token::Ident("name".into()),
                Token::Eq,
                Token::StringLit("Alice".into()),
                Token::Return,
                Token::Ident("m".into()),
                Token::Dot,
                Token::Ident("name".into()),
                Token::Limit,
                Token::IntLit(10),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_line_comments() {
        let tokens = tok("MATCH // this is a comment\nRETURN");
        assert_eq!(tokens, vec![Token::Match, Token::Return, Token::Eof]);
    }

    #[test]
    fn test_number_before_dotdot() {
        // 1..5 should be IntLit(1), DotDot, IntLit(5)
        let tokens = tok("1..5");
        assert_eq!(
            tokens,
            vec![
                Token::IntLit(1),
                Token::DotDot,
                Token::IntLit(5),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn test_spans() {
        let spanned = Lexer::tokenize("MATCH x").unwrap();
        assert_eq!(spanned[0].span, (0, 5));
        assert_eq!(spanned[1].span, (6, 7));
    }

    #[test]
    fn test_braces_and_star() {
        let tokens = tok("{ * }");
        assert_eq!(
            tokens,
            vec![Token::LBrace, Token::Star, Token::RBrace, Token::Eof]
        );
    }

    #[test]
    fn test_boolean_keywords() {
        let tokens = tok("TRUE false");
        assert_eq!(tokens, vec![Token::True, Token::False, Token::Eof]);
    }
}
