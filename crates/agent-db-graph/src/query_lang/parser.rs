use super::ast::*;
use super::lexer::Lexer;
use super::token::{Spanned, Token};
use super::types::QueryError;

const MAX_PARSE_DEPTH: usize = 64;

pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    depth: usize,
}

impl Parser {
    pub fn parse(input: &str) -> Result<Query, QueryError> {
        let tokens = Lexer::tokenize(input).map_err(|e| QueryError::ParseError {
            message: e.message,
            position: e.position,
        })?;
        let mut parser = Parser {
            tokens,
            pos: 0,
            depth: 0,
        };
        parser.parse_query()
    }

    /// Parse a full statement (SUBSCRIBE, UNSUBSCRIBE, or regular query).
    pub fn parse_statement(input: &str) -> Result<Statement, QueryError> {
        let tokens = Lexer::tokenize(input).map_err(|e| QueryError::ParseError {
            message: e.message,
            position: e.position,
        })?;
        let mut parser = Parser {
            tokens,
            pos: 0,
            depth: 0,
        };

        if parser.at(&Token::Subscribe) {
            parser.advance(); // consume SUBSCRIBE
            let query = parser.parse_query()?;
            Ok(Statement::Subscribe(query))
        } else if parser.at(&Token::Unsubscribe) {
            parser.advance(); // consume UNSUBSCRIBE
                              // Expect an integer literal (subscription ID).
            match parser.peek() {
                Token::IntLit(id) => {
                    let id = *id as u64;
                    parser.advance();
                    if !parser.at(&Token::Eof) {
                        return Err(
                            parser.error("expected end of input after UNSUBSCRIBE <id>".into())
                        );
                    }
                    Ok(Statement::Unsubscribe(id))
                },
                other => Err(parser.error(format!(
                    "expected subscription ID after UNSUBSCRIBE, got {:?}",
                    other
                ))),
            }
        } else if parser.at(&Token::Create) {
            // Peek next token to distinguish CREATE TABLE vs CREATE [UNIQUE] INDEX.
            // `INDEX`, `UNIQUE`, and `TABLE` are soft keywords (Token::Ident).
            let is_ident_kw = |tok: Option<&Token>, kw: &str| -> bool {
                matches!(tok, Some(Token::Ident(s)) if s.eq_ignore_ascii_case(kw))
            };
            let next = parser.tokens.get(parser.pos + 1).map(|s| &s.token);
            let next2 = parser.tokens.get(parser.pos + 2).map(|s| &s.token);
            if is_ident_kw(next, "INDEX")
                || (is_ident_kw(next, "UNIQUE") && is_ident_kw(next2, "INDEX"))
            {
                parser.parse_create_index().map(Statement::CreateIndex)
            } else {
                parser.parse_create_table().map(Statement::CreateTable)
            }
        } else if parser.at(&Token::Drop) {
            parser.parse_drop_table()
        } else if parser.at(&Token::Insert) {
            parser.parse_insert_into().map(Statement::InsertInto)
        } else if parser.at(&Token::Update) {
            parser.parse_update_table().map(Statement::UpdateTable)
        } else if parser.at(&Token::Delete) {
            parser.parse_delete_from().map(Statement::DeleteFrom)
        } else if parser.at(&Token::Alter) {
            parser.parse_alter_table().map(Statement::AlterTable)
        } else {
            let query = parser.parse_query()?;
            Ok(Statement::Query(query))
        }
    }

    // ── Top-level ───────────────────────────────────────────────────────

    fn parse_query(&mut self) -> Result<Query, QueryError> {
        // Determine query shape: MATCH or FROM
        let (match_clauses, from_table) = if self.at(&Token::From) {
            self.advance(); // consume FROM
            let table_name = self.expect_ident()?;
            // Optional alias: `FROM table AS alias`, or `FROM table alias`.
            // The bare-alias form must not swallow a following clause keyword.
            // Clause-delimiter keywords retain their specialized tokens, so a
            // negative check against the known set is still exhaustive.
            let alias = if self.at(&Token::As) {
                self.advance();
                Some(self.expect_ident()?)
            } else if matches!(self.peek(), Token::Ident(_))
                && !self.at(&Token::Join)
                && !self.at(&Token::Left)
                && !self.at(&Token::Inner)
                && !self.at(&Token::Where)
                && !self.at(&Token::Return)
                && !self.at(&Token::When)
                && !self.at(&Token::Order)
                && !self.at(&Token::Group)
                && !self.at(&Token::Limit)
            {
                Some(self.expect_ident()?)
            } else {
                None
            };
            (
                Vec::new(),
                Some(TableRef {
                    name: table_name,
                    alias,
                }),
            )
        } else {
            (self.parse_match()?, None)
        };

        // Parse zero or more JOIN clauses
        let joins = self.parse_joins()?;

        let when = self.parse_when()?;
        let as_of = self.parse_as_of()?;
        let where_clause = self.parse_where()?;

        // Parse optional GROUP BY
        let group_by = self.parse_group_by()?;

        // Parse optional HAVING (post-aggregation filter)
        let having = if self.at(&Token::Having) {
            self.advance();
            Some(self.parse_or_expr()?)
        } else {
            None
        };

        let returns = self.parse_return()?;
        let order_by = self.parse_order_by()?;
        let limit = self.parse_limit()?;

        if !self.at(&Token::Eof) {
            return Err(self.error(format!("unexpected token {:?}", self.peek())));
        }

        Ok(Query {
            match_clauses,
            from_table,
            joins,
            when,
            as_of,
            where_clause,
            group_by,
            having,
            returns,
            order_by,
            limit,
        })
    }

    // ── MATCH ───────────────────────────────────────────────────────────

    fn parse_match(&mut self) -> Result<Vec<Pattern>, QueryError> {
        self.expect(&Token::Match)?;
        let mut patterns = vec![self.parse_pattern()?];
        while self.at(&Token::Comma) {
            self.advance();
            patterns.push(self.parse_pattern()?);
        }
        Ok(patterns)
    }

    fn parse_pattern(&mut self) -> Result<Pattern, QueryError> {
        let mut elements = Vec::new();
        let node = self.parse_node_pattern()?;
        elements.push(PatternElement::Node(node));

        while let Some((edge, dir, node)) = self.parse_edge_and_node()? {
            elements.push(PatternElement::Edge(edge, dir));
            elements.push(PatternElement::Node(node));
        }

        Ok(Pattern { elements })
    }

    fn parse_node_pattern(&mut self) -> Result<NodePattern, QueryError> {
        self.expect(&Token::LParen)?;

        let var = if matches!(self.peek(), Token::Ident(_)) {
            let name = self.expect_ident()?;
            Some(name)
        } else {
            None
        };

        let mut labels = Vec::new();
        while self.at(&Token::Colon) {
            self.advance();
            labels.push(self.expect_ident()?);
        }

        let props = if self.at(&Token::LBrace) {
            self.parse_props()?
        } else {
            Vec::new()
        };

        self.expect(&Token::RParen)?;

        Ok(NodePattern { var, labels, props })
    }

    fn parse_edge_and_node(
        &mut self,
    ) -> Result<Option<(EdgePattern, Direction, NodePattern)>, QueryError> {
        if self.at(&Token::Dash) {
            // Outgoing: -[...]->(node)
            self.advance(); // consume Dash
            self.expect(&Token::LBracket)?;
            let edge = self.parse_edge_inner()?;
            self.expect(&Token::RBracket)?;
            self.expect(&Token::Arrow)?; // ->
            let node = self.parse_node_pattern()?;
            Ok(Some((edge, Direction::Out, node)))
        } else if self.at(&Token::BackArrow) {
            // Incoming: <-[...]-(node)
            self.advance(); // consume BackArrow (<-)
            self.expect(&Token::LBracket)?;
            let edge = self.parse_edge_inner()?;
            self.expect(&Token::RBracket)?;
            self.expect(&Token::Dash)?;
            let node = self.parse_node_pattern()?;
            Ok(Some((edge, Direction::In, node)))
        } else {
            Ok(None)
        }
    }

    fn parse_edge_inner(&mut self) -> Result<EdgePattern, QueryError> {
        let mut var = None;
        let mut edge_type = None;
        let mut range = None;
        let mut props = Vec::new();

        // Parse optional var and/or edge_type
        if matches!(self.peek(), Token::Ident(_)) {
            var = Some(self.expect_ident()?);
            if self.at(&Token::Colon) {
                self.advance();
                edge_type = Some(self.expect_ident()?);
            }
        } else if self.at(&Token::Colon) {
            self.advance();
            edge_type = Some(self.expect_ident()?);
        }

        // Parse optional range: * [..max]
        if self.at(&Token::Star) {
            self.advance();
            if self.at(&Token::DotDot) {
                self.advance();
                let max = self.expect_int()? as u32;
                range = Some((1, Some(max)));
            } else {
                range = Some((1, None));
            }
        }

        // Parse optional props
        if self.at(&Token::LBrace) {
            props = self.parse_props()?;
        }

        Ok(EdgePattern {
            var,
            edge_type,
            range,
            props,
        })
    }

    // ── WHEN ────────────────────────────────────────────────────────────

    fn parse_when(&mut self) -> Result<Option<WhenClause>, QueryError> {
        if !self.at(&Token::When) {
            return Ok(None);
        }
        self.advance();

        // WHEN ALL
        if self.consume_keyword("ALL") {
            return Ok(Some(WhenClause::All));
        }

        // WHEN LAST "duration"
        if self.consume_keyword("LAST") {
            match self.peek().clone() {
                Token::StringLit(s) => {
                    let val = s.clone();
                    self.advance();
                    return Ok(Some(WhenClause::Last(val)));
                },
                _ => return Err(self.error("expected string literal after LAST".into())),
            }
        }

        // WHEN expr [TO expr]
        let expr1 = self.parse_expr()?;
        if self.consume_keyword("TO") {
            let expr2 = self.parse_expr()?;
            Ok(Some(WhenClause::Range(expr1, expr2)))
        } else {
            Ok(Some(WhenClause::PointInTime(expr1)))
        }
    }

    // ── AS OF ──────────────────────────────────────────────────────────

    fn parse_as_of(&mut self) -> Result<Option<Expr>, QueryError> {
        if !self.at(&Token::As) {
            return Ok(None);
        }
        // Peek ahead: must be AS OF, not AS (alias keyword)
        if self.tokens.get(self.pos + 1).map(|s| &s.token) != Some(&Token::Of) {
            return Ok(None);
        }
        self.advance(); // consume AS
        self.advance(); // consume OF
        let expr = self.parse_expr()?;
        Ok(Some(expr))
    }

    // ── WHERE ───────────────────────────────────────────────────────────

    fn parse_where(&mut self) -> Result<Option<BoolExpr>, QueryError> {
        if !self.at(&Token::Where) {
            return Ok(None);
        }
        self.advance();
        let expr = self.parse_or_expr()?;
        Ok(Some(expr))
    }

    // ── GROUP BY ────────────────────────────────────────────────────────

    fn parse_group_by(&mut self) -> Result<Vec<Expr>, QueryError> {
        if !self.at(&Token::Group) {
            return Ok(Vec::new());
        }
        self.advance(); // consume GROUP
        self.expect(&Token::By)?;

        let mut exprs = vec![self.parse_expr()?];
        while self.at(&Token::Comma) {
            self.advance();
            exprs.push(self.parse_expr()?);
        }
        Ok(exprs)
    }

    fn parse_or_expr(&mut self) -> Result<BoolExpr, QueryError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(self.error("expression nesting depth exceeds limit".into()));
        }
        let mut left = self.parse_and_expr()?;
        while self.at(&Token::Or) {
            self.advance();
            let right = self.parse_and_expr()?;
            left = BoolExpr::Or(Box::new(left), Box::new(right));
        }
        self.depth -= 1;
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<BoolExpr, QueryError> {
        let mut left = self.parse_not_expr()?;
        while self.at(&Token::And) {
            self.advance();
            let right = self.parse_not_expr()?;
            left = BoolExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<BoolExpr, QueryError> {
        if self.at(&Token::Not) {
            self.advance();
            self.depth += 1;
            if self.depth > MAX_PARSE_DEPTH {
                return Err(self.error("expression nesting depth exceeds limit".into()));
            }
            let inner = self.parse_not_expr()?;
            self.depth -= 1;
            return Ok(BoolExpr::Not(Box::new(inner)));
        }
        if self.at(&Token::LParen) {
            self.advance();
            let inner = self.parse_or_expr()?;
            self.expect(&Token::RParen)?;
            return Ok(BoolExpr::Paren(Box::new(inner)));
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<BoolExpr, QueryError> {
        let left = self.parse_expr()?;

        // IS NULL / IS NOT NULL
        if self.at(&Token::Is) {
            self.advance();
            if self.at(&Token::Not) {
                self.advance();
                self.expect(&Token::Null)?;
                return Ok(BoolExpr::IsNotNull(left));
            }
            self.expect(&Token::Null)?;
            return Ok(BoolExpr::IsNull(left));
        }

        // NOT IN (...)
        if self.at(&Token::Not) {
            let next = self.tokens.get(self.pos + 1).map(|s| &s.token);
            if next == Some(&Token::In) {
                self.advance(); // consume NOT
                self.advance(); // consume IN
                let values = self.parse_expr_list()?;
                return Ok(BoolExpr::NotIn(left, values));
            }
        }

        // IN (...)
        if self.at(&Token::In) {
            self.advance();
            let values = self.parse_expr_list()?;
            return Ok(BoolExpr::In(left, values));
        }

        // BETWEEN low AND high
        if self.at(&Token::Between) {
            self.advance();
            let low = self.parse_expr()?;
            self.expect(&Token::And)?;
            let high = self.parse_expr()?;
            return Ok(BoolExpr::Between(left, low, high));
        }

        // LIKE "pattern"
        if self.at(&Token::Like) {
            self.advance();
            match self.peek().clone() {
                Token::StringLit(s) => {
                    let pattern = s.clone();
                    self.advance();
                    return Ok(BoolExpr::Like(left, pattern));
                },
                _ => return Err(self.error("expected string literal after LIKE".into())),
            }
        }

        let op = match self.peek() {
            Token::Eq => CompOp::Eq,
            Token::Neq => CompOp::Neq,
            Token::Lt => CompOp::Lt,
            Token::Gt => CompOp::Gt,
            Token::Lte => CompOp::Lte,
            Token::Gte => CompOp::Gte,
            Token::Contains => CompOp::Contains,
            Token::StartsWith => CompOp::StartsWith,
            _ => {
                // If the expression is a function call with no comparison operator
                // following, treat it as a boolean-returning function predicate.
                if let Expr::FuncCall(name, args) = left {
                    return Ok(BoolExpr::FuncPredicate(name, args));
                }
                return Err(self.error(format!(
                    "expected comparison operator, found {:?}",
                    self.peek()
                )));
            },
        };
        self.advance();
        let right = self.parse_expr()?;
        Ok(BoolExpr::Comparison(left, op, right))
    }

    /// Parse parenthesized expression list: (expr, expr, ...)
    fn parse_expr_list(&mut self) -> Result<Vec<Expr>, QueryError> {
        self.expect(&Token::LParen)?;
        let mut exprs = vec![self.parse_expr()?];
        while self.at(&Token::Comma) {
            self.advance();
            exprs.push(self.parse_expr()?);
        }
        self.expect(&Token::RParen)?;
        Ok(exprs)
    }

    // ── RETURN ──────────────────────────────────────────────────────────

    fn parse_return(&mut self) -> Result<Vec<ReturnItem>, QueryError> {
        self.expect(&Token::Return)?;
        let mut items = vec![self.parse_return_item()?];
        while self.at(&Token::Comma) {
            self.advance();
            items.push(self.parse_return_item()?);
        }
        Ok(items)
    }

    fn parse_return_item(&mut self) -> Result<ReturnItem, QueryError> {
        let distinct = if self.at(&Token::Distinct) {
            self.advance();
            true
        } else {
            false
        };

        let expr = self.parse_expr()?;

        let alias = if self.at(&Token::As) {
            self.advance();
            Some(self.expect_ident()?)
        } else {
            None
        };

        Ok(ReturnItem {
            expr,
            alias,
            distinct,
        })
    }

    // ── ORDER BY ────────────────────────────────────────────────────────

    fn parse_order_by(&mut self) -> Result<Vec<OrderItem>, QueryError> {
        if !self.at(&Token::Order) {
            return Ok(Vec::new());
        }
        self.advance();
        self.expect(&Token::By)?;

        let mut items = vec![self.parse_order_item()?];
        while self.at(&Token::Comma) {
            self.advance();
            items.push(self.parse_order_item()?);
        }
        Ok(items)
    }

    fn parse_order_item(&mut self) -> Result<OrderItem, QueryError> {
        let expr = self.parse_expr()?;
        let descending = if self.at(&Token::Desc) {
            self.advance();
            true
        } else {
            if self.at(&Token::Asc) {
                self.advance();
            }
            false
        };
        Ok(OrderItem { expr, descending })
    }

    // ── LIMIT ───────────────────────────────────────────────────────────

    fn parse_limit(&mut self) -> Result<Option<u64>, QueryError> {
        if !self.at(&Token::Limit) {
            return Ok(None);
        }
        self.advance();
        let val = self.expect_int()?;
        if val < 0 {
            return Err(self.error("LIMIT value must be non-negative".into()));
        }
        Ok(Some(val as u64))
    }

    // ── Expressions ─────────────────────────────────────────────────────

    fn parse_expr(&mut self) -> Result<Expr, QueryError> {
        match self.peek().clone() {
            Token::Star => {
                self.advance();
                Ok(Expr::Star)
            },
            Token::StringLit(s) => {
                let val = s.clone();
                self.advance();
                Ok(Expr::Literal(Literal::String(val)))
            },
            Token::IntLit(i) => {
                let val = i;
                self.advance();
                Ok(Expr::Literal(Literal::Int(val)))
            },
            Token::FloatLit(f) => {
                let val = f;
                self.advance();
                Ok(Expr::Literal(Literal::Float(val)))
            },
            Token::True => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(true)))
            },
            Token::False => {
                self.advance();
                Ok(Expr::Literal(Literal::Bool(false)))
            },
            Token::Null => {
                self.advance();
                Ok(Expr::Literal(Literal::Null))
            },
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();

                // func(args)
                if self.at(&Token::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    if !self.at(&Token::RParen) {
                        args.push(self.parse_expr()?);
                        while self.at(&Token::Comma) {
                            self.advance();
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::FuncCall(name, args));
                }

                // var.prop
                if self.at(&Token::Dot) {
                    self.advance();
                    let prop = self.expect_ident()?;
                    return Ok(Expr::Property(name, prop));
                }

                // bare var
                Ok(Expr::Var(name))
            },
            _ => Err(self.error(format!("expected expression, found {:?}", self.peek()))),
        }
    }

    // ── Props ───────────────────────────────────────────────────────────

    fn parse_props(&mut self) -> Result<Vec<(String, Literal)>, QueryError> {
        self.expect(&Token::LBrace)?;
        let mut props = Vec::new();
        if !self.at(&Token::RBrace) {
            let key = self.expect_ident()?;
            self.expect(&Token::Colon)?;
            let val = self.parse_literal()?;
            props.push((key, val));

            while self.at(&Token::Comma) {
                self.advance();
                let key = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                let val = self.parse_literal()?;
                props.push((key, val));
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(props)
    }

    fn parse_literal(&mut self) -> Result<Literal, QueryError> {
        match self.peek().clone() {
            Token::StringLit(s) => {
                let val = s.clone();
                self.advance();
                Ok(Literal::String(val))
            },
            Token::IntLit(i) => {
                let val = i;
                self.advance();
                Ok(Literal::Int(val))
            },
            Token::FloatLit(f) => {
                let val = f;
                self.advance();
                Ok(Literal::Float(val))
            },
            Token::True => {
                self.advance();
                Ok(Literal::Bool(true))
            },
            Token::False => {
                self.advance();
                Ok(Literal::Bool(false))
            },
            Token::Null => {
                self.advance();
                Ok(Literal::Null)
            },
            // NODE(id) — graph node reference for NodeRef columns
            Token::Ident(ref s) if s.eq_ignore_ascii_case("NODE") => {
                self.advance();
                self.expect(&Token::LParen)?;
                let id = match self.peek().clone() {
                    Token::IntLit(i) if i >= 0 => {
                        self.advance();
                        i as u64
                    },
                    _ => {
                        return Err(self.error(format!(
                            "NODE() requires a non-negative integer id, found {:?}",
                            self.peek()
                        )));
                    },
                };
                self.expect(&Token::RParen)?;
                Ok(Literal::NodeRef(id))
            },
            _ => Err(self.error(format!("expected literal, found {:?}", self.peek()))),
        }
    }

    // ── JOIN ──────────────────────────────────────────────────────────

    fn parse_joins(&mut self) -> Result<Vec<JoinClause>, QueryError> {
        let mut joins = Vec::new();
        loop {
            // Determine join type: LEFT [JOIN], INNER [JOIN], or bare JOIN
            let join_type = if self.at(&Token::Left) {
                self.advance();
                if self.at(&Token::Join) {
                    self.advance();
                }
                JoinType::Left
            } else if self.at(&Token::Inner) {
                self.advance();
                if self.at(&Token::Join) {
                    self.advance();
                }
                JoinType::Inner
            } else if self.at(&Token::Join) {
                self.advance();
                JoinType::Inner // bare JOIN is INNER
            } else {
                break;
            };

            let table = self.expect_ident()?;

            // Optional alias: JOIN table AS alias, or JOIN table alias
            let alias = if self.at(&Token::As) {
                self.advance();
                Some(self.expect_ident()?)
            } else if matches!(self.peek(), Token::Ident(_)) && !self.at(&Token::On) {
                Some(self.expect_ident()?)
            } else {
                None
            };

            self.expect(&Token::On)?;
            let (on_left, on_right) = self.parse_join_condition()?;
            joins.push(JoinClause {
                join_type,
                table,
                alias,
                on_left,
                on_right,
            });
        }
        Ok(joins)
    }

    /// Parse: left_side = right_side
    /// Each side is either table.column or a bare identifier (graph var).
    fn parse_join_condition(&mut self) -> Result<(JoinSide, JoinSide), QueryError> {
        let left = self.parse_join_side()?;
        self.expect(&Token::Eq)?;
        let right = self.parse_join_side()?;
        Ok((left, right))
    }

    fn parse_join_side(&mut self) -> Result<JoinSide, QueryError> {
        let name = self.expect_ident()?;
        if self.at(&Token::Dot) {
            self.advance();
            let col = self.expect_ident()?;
            Ok(JoinSide::TableColumn {
                table: name,
                column: col,
            })
        } else {
            Ok(JoinSide::GraphVar(name))
        }
    }

    // ── DDL ──────────────────────────────────────────────────────────

    /// CREATE TABLE name (col_defs [, constraints])
    fn parse_create_table(&mut self) -> Result<CreateTableStmt, QueryError> {
        self.expect(&Token::Create)?;
        self.expect_keyword("TABLE")?;
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;

        let mut columns = Vec::new();
        let mut constraints = Vec::new();

        loop {
            if self.at(&Token::RParen) {
                break;
            }

            // Table-level constraints require two-token lookahead so that
            // `CREATE TABLE t (primary INT)` parses `primary` as a column name
            // rather than committing to a broken PRIMARY KEY constraint:
            //   `PRIMARY KEY (...)`  → table constraint
            //   `UNIQUE (...)`       → table constraint
            //   `primary INT`        → column definition (column named `primary`)
            //   `unique VARCHAR`     → column definition (column named `unique`)
            let next_tok = self.tokens.get(self.pos + 1).map(|s| &s.token);
            let is_primary_constraint = self.at_keyword("PRIMARY")
                && matches!(next_tok, Some(Token::Ident(s)) if s.eq_ignore_ascii_case("KEY"));
            let is_unique_constraint =
                self.at_keyword("UNIQUE") && matches!(next_tok, Some(Token::LParen));

            if is_primary_constraint {
                self.expect_keyword("PRIMARY")?;
                self.expect_keyword("KEY")?;
                self.expect(&Token::LParen)?;
                let cols = self.parse_ident_list()?;
                self.expect(&Token::RParen)?;
                constraints.push(ConstraintAst {
                    kind: ConstraintKind::PrimaryKey(cols),
                });
            } else if is_unique_constraint {
                self.expect_keyword("UNIQUE")?;
                self.expect(&Token::LParen)?;
                let cols = self.parse_ident_list()?;
                self.expect(&Token::RParen)?;
                constraints.push(ConstraintAst {
                    kind: ConstraintKind::Unique(cols),
                });
            } else {
                // Column definition
                let col = self.parse_column_def()?;

                // Inline PRIMARY KEY → extract to constraint
                if col.is_primary_key {
                    constraints.push(ConstraintAst {
                        kind: ConstraintKind::PrimaryKey(vec![col.name.clone()]),
                    });
                }

                columns.push(col);
            }

            if !self.at(&Token::Comma) {
                break;
            }
            self.advance(); // consume comma
        }

        self.expect(&Token::RParen)?;

        Ok(CreateTableStmt {
            name,
            columns,
            constraints,
        })
    }

    /// Parse a single column def: name Type [NOT NULL] [PRIMARY KEY] [DEFAULT literal] [AUTOINCREMENT] [REFERENCES GRAPH]
    fn parse_column_def(&mut self) -> Result<ColumnDefAst, QueryError> {
        let name = self.expect_ident()?;
        let col_type = self.expect_ident()?;
        let mut nullable = true;
        let mut is_primary_key = false;
        let mut default_value = None;
        let mut autoincrement = false;

        // Optional modifiers (any order). Modifier keywords are soft keywords
        // (`PRIMARY`, `DEFAULT`, `AUTOINCREMENT`, `REFERENCES`) lexed as Ident.
        // Inside the modifier loop there is no ambiguity with column names —
        // a column def ends at `,` or `)` — so a direct `at_keyword` check is
        // sufficient.
        loop {
            if self.at(&Token::Not) {
                self.advance();
                self.expect(&Token::Null)?;
                nullable = false;
            } else if self.at_keyword("PRIMARY") {
                self.advance();
                self.expect_keyword("KEY")?;
                is_primary_key = true;
                nullable = false; // PKs are implicitly NOT NULL
            } else if self.at_keyword("DEFAULT") {
                self.advance();
                default_value = Some(self.parse_literal()?);
            } else if self.at_keyword("AUTOINCREMENT") {
                self.advance();
                autoincrement = true;
                nullable = false; // auto-increment columns are implicitly NOT NULL
            } else if self.at_keyword("REFERENCES") {
                self.advance();
                self.expect_keyword("GRAPH")?;
            } else {
                break;
            }
        }

        Ok(ColumnDefAst {
            name,
            col_type,
            nullable,
            is_primary_key,
            default_value,
            autoincrement,
        })
    }

    /// CREATE [UNIQUE] INDEX name ON table (col1, col2, ...)
    fn parse_create_index(&mut self) -> Result<CreateIndexStmt, QueryError> {
        self.expect(&Token::Create)?;
        let unique = self.consume_keyword("UNIQUE");
        self.expect_keyword("INDEX")?;
        let index_name = self.expect_ident()?;
        self.expect(&Token::On)?;
        let table = self.expect_ident()?;
        self.expect(&Token::LParen)?;
        let columns = self.parse_ident_list()?;
        self.expect(&Token::RParen)?;
        Ok(CreateIndexStmt {
            index_name,
            table,
            columns,
            unique,
        })
    }

    /// DROP TABLE name
    fn parse_drop_table(&mut self) -> Result<Statement, QueryError> {
        self.expect(&Token::Drop)?;
        self.expect_keyword("TABLE")?;
        let name = self.expect_ident()?;
        Ok(Statement::DropTable(name))
    }

    /// ALTER TABLE name ADD [COLUMN] col_def [, ADD [COLUMN] col_def ...]
    fn parse_alter_table(&mut self) -> Result<AlterTableStmt, QueryError> {
        self.expect(&Token::Alter)?;
        self.expect_keyword("TABLE")?;
        let table = self.expect_ident()?;

        let mut add_columns = Vec::new();
        loop {
            self.expect_keyword("ADD")?;
            // Optional COLUMN keyword
            self.consume_keyword("COLUMN");
            add_columns.push(self.parse_column_def()?);
            if !self.at(&Token::Comma) {
                break;
            }
            self.advance();
        }

        Ok(AlterTableStmt { table, add_columns })
    }

    // ── DML ──────────────────────────────────────────────────────────

    /// INSERT INTO table [(columns)] VALUES (vals) [, (vals)]*
    fn parse_insert_into(&mut self) -> Result<InsertStmt, QueryError> {
        self.expect(&Token::Insert)?;
        self.expect_keyword("INTO")?;
        let table = self.expect_ident()?;

        // Optional column list
        let columns = if self.at(&Token::LParen) {
            // Could be column list or VALUES.
            // Peek ahead: if next token after LParen is an ident (not a literal), it's columns.
            let saved = self.pos;
            self.advance(); // consume (
            if matches!(self.peek(), Token::Ident(_)) {
                let cols = self.parse_ident_list()?;
                self.expect(&Token::RParen)?;
                Some(cols)
            } else {
                // Restore — it's the VALUES parens
                self.pos = saved;
                None
            }
        } else {
            None
        };

        self.expect_keyword("VALUES")?;

        let mut rows = Vec::new();
        loop {
            self.expect(&Token::LParen)?;
            let mut vals = Vec::new();
            if !self.at(&Token::RParen) {
                vals.push(self.parse_literal()?);
                while self.at(&Token::Comma) {
                    self.advance();
                    vals.push(self.parse_literal()?);
                }
            }
            self.expect(&Token::RParen)?;
            rows.push(vals);

            if !self.at(&Token::Comma) {
                break;
            }
            self.advance(); // consume comma between row groups
        }

        Ok(InsertStmt {
            table,
            columns,
            rows,
        })
    }

    /// UPDATE table SET col = expr [, col = expr]* WHERE condition
    fn parse_update_table(&mut self) -> Result<UpdateStmt, QueryError> {
        self.expect(&Token::Update)?;
        let table = self.expect_ident()?;
        self.expect_keyword("SET")?;

        let mut assignments = Vec::new();
        loop {
            let col = self.expect_ident()?;
            self.expect(&Token::Eq)?;
            let val = self.parse_expr()?;
            assignments.push((col, val));
            if !self.at(&Token::Comma) {
                break;
            }
            self.advance();
        }

        if !self.at(&Token::Where) {
            return Err(self.error("UPDATE requires a WHERE clause".into()));
        }
        self.advance();
        let where_clause = self.parse_or_expr()?;

        Ok(UpdateStmt {
            table,
            assignments,
            where_clause,
        })
    }

    /// DELETE FROM table WHERE condition
    fn parse_delete_from(&mut self) -> Result<DeleteStmt, QueryError> {
        self.expect(&Token::Delete)?;
        self.expect(&Token::From)?;
        let table = self.expect_ident()?;

        if !self.at(&Token::Where) {
            return Err(self.error("DELETE FROM requires a WHERE clause".into()));
        }
        self.advance();
        let where_clause = self.parse_or_expr()?;

        Ok(DeleteStmt {
            table,
            where_clause,
        })
    }

    /// Parse comma-separated identifier list.
    fn parse_ident_list(&mut self) -> Result<Vec<String>, QueryError> {
        let mut names = vec![self.expect_ident()?];
        while self.at(&Token::Comma) {
            self.advance();
            names.push(self.expect_ident()?);
        }
        Ok(names)
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .map(|s| &s.token)
            .unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Spanned {
        let s = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        s
    }

    fn expect(&mut self, expected: &Token) -> Result<(), QueryError> {
        if std::mem::discriminant(self.peek()) == std::mem::discriminant(expected) {
            self.advance();
            Ok(())
        } else {
            Err(self.error(format!("expected {:?}, found {:?}", expected, self.peek())))
        }
    }

    fn at(&self, token: &Token) -> bool {
        std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
    }

    fn current_pos(&self) -> usize {
        self.tokens.get(self.pos).map(|s| s.span.0).unwrap_or(0)
    }

    fn expect_ident(&mut self) -> Result<String, QueryError> {
        match self.peek().clone() {
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();
                Ok(name)
            },
            other => Err(self.error(format!("expected identifier, found {:?}", other))),
        }
    }

    // ── Soft-keyword helpers ────────────────────────────────────────────
    //
    // Most MinnsQL keywords (`TABLE`, `PRIMARY`, `KEY`, `VALUES`, `SET`, etc.)
    // lex as `Token::Ident(String)` and are recognized by case-insensitive
    // string comparison at the specific parser sites where they have
    // structural meaning. These helpers are the canonical way to do that
    // recognition, replacing the hand-written `Token::X` matches that used
    // to pepper the DDL and clause-parsing code.

    /// True if the current token is an `Ident` matching `kw` case-insensitively.
    fn at_keyword(&self, kw: &str) -> bool {
        matches!(self.peek(), Token::Ident(s) if s.eq_ignore_ascii_case(kw))
    }

    /// If the current token is `Ident(kw)` (case-insensitive), consume it and
    /// return `true`. Otherwise leave the position alone and return `false`.
    fn consume_keyword(&mut self, kw: &str) -> bool {
        if self.at_keyword(kw) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Require the current token to be `Ident(kw)` (case-insensitive), consume
    /// it, and return Ok. Error with a position-bearing parse error otherwise.
    fn expect_keyword(&mut self, kw: &str) -> Result<(), QueryError> {
        if self.consume_keyword(kw) {
            Ok(())
        } else {
            Err(self.error(format!("expected `{}`, found {:?}", kw, self.peek())))
        }
    }

    fn expect_int(&mut self) -> Result<i64, QueryError> {
        match self.peek().clone() {
            Token::IntLit(i) => {
                self.advance();
                Ok(i)
            },
            other => Err(self.error(format!("expected integer, found {:?}", other))),
        }
    }

    fn error(&self, message: String) -> QueryError {
        QueryError::ParseError {
            message,
            position: self.current_pos(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Simple MATCH ... RETURN query
    #[test]
    fn test_simple_match_return() {
        let q = Parser::parse("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(q.match_clauses.len(), 1);
        let pat = &q.match_clauses[0];
        assert_eq!(pat.elements.len(), 1);
        match &pat.elements[0] {
            PatternElement::Node(np) => {
                assert_eq!(np.var.as_deref(), Some("n"));
                assert_eq!(np.labels, vec!["Person"]);
                assert!(np.props.is_empty());
            },
            _ => panic!("expected node pattern"),
        }
        assert_eq!(q.returns.len(), 1);
        match &q.returns[0].expr {
            Expr::Var(v) => assert_eq!(v, "n"),
            _ => panic!("expected Var"),
        }
        assert!(q.when.is_none());
        assert!(q.where_clause.is_none());
        assert!(q.order_by.is_empty());
        assert!(q.limit.is_none());
    }

    // 2. MATCH with outgoing edge pattern
    #[test]
    fn test_match_with_edge() {
        let q = Parser::parse(r#"MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name"#)
            .unwrap();
        let pat = &q.match_clauses[0];
        assert_eq!(pat.elements.len(), 3);

        match &pat.elements[0] {
            PatternElement::Node(n) => {
                assert_eq!(n.var.as_deref(), Some("a"));
                assert_eq!(n.labels, vec!["Person"]);
            },
            _ => panic!("expected node"),
        }
        match &pat.elements[1] {
            PatternElement::Edge(e, d) => {
                assert_eq!(e.var.as_deref(), Some("r"));
                assert_eq!(e.edge_type.as_deref(), Some("KNOWS"));
                assert_eq!(*d, Direction::Out);
            },
            _ => panic!("expected edge"),
        }
        match &pat.elements[2] {
            PatternElement::Node(n) => {
                assert_eq!(n.var.as_deref(), Some("b"));
            },
            _ => panic!("expected node"),
        }
        assert_eq!(q.returns.len(), 2);
    }

    // 3. WHEN clauses
    #[test]
    fn test_when_all() {
        let q = Parser::parse("MATCH (n) WHEN ALL RETURN n").unwrap();
        assert!(matches!(q.when, Some(WhenClause::All)));
    }

    #[test]
    fn test_when_last() {
        let q = Parser::parse(r#"MATCH (n) WHEN LAST "30d" RETURN n"#).unwrap();
        match &q.when {
            Some(WhenClause::Last(d)) => assert_eq!(d, "30d"),
            other => panic!("expected WhenClause::Last, got {:?}", other),
        }
    }

    #[test]
    fn test_when_point_in_time() {
        let q = Parser::parse(r#"MATCH (n) WHEN "2026-01-01" RETURN n"#).unwrap();
        match &q.when {
            Some(WhenClause::PointInTime(Expr::Literal(Literal::String(s)))) => {
                assert_eq!(s, "2026-01-01");
            },
            other => panic!("expected PointInTime, got {:?}", other),
        }
    }

    #[test]
    fn test_when_range() {
        let q = Parser::parse(r#"MATCH (n) WHEN "2026-01-01" TO "2026-02-01" RETURN n"#).unwrap();
        match &q.when {
            Some(WhenClause::Range(
                Expr::Literal(Literal::String(a)),
                Expr::Literal(Literal::String(b)),
            )) => {
                assert_eq!(a, "2026-01-01");
                assert_eq!(b, "2026-02-01");
            },
            other => panic!("expected Range, got {:?}", other),
        }
    }

    // 4. WHERE with AND/OR
    #[test]
    fn test_where_and_or() {
        let q = Parser::parse(
            r#"MATCH (n) WHERE n.age > 20 AND n.name = "Alice" OR n.active = true RETURN n"#,
        )
        .unwrap();
        // Precedence: AND binds tighter, so (age>20 AND name="Alice") OR active=true
        match &q.where_clause {
            Some(BoolExpr::Or(left, _right)) => {
                assert!(matches!(left.as_ref(), BoolExpr::And(_, _)));
            },
            other => panic!("expected Or at top level, got {:?}", other),
        }
    }

    // 5. ORDER BY and LIMIT
    #[test]
    fn test_order_by_and_limit() {
        let q = Parser::parse(
            "MATCH (n:Person) RETURN n.name ORDER BY n.name ASC, n.age DESC LIMIT 25",
        )
        .unwrap();
        assert_eq!(q.order_by.len(), 2);
        assert!(!q.order_by[0].descending);
        assert!(q.order_by[1].descending);
        assert_eq!(q.limit, Some(25));
    }

    // 6. Multiple comma-separated patterns
    #[test]
    fn test_multiple_patterns() {
        let q = Parser::parse(
            "MATCH (a:Person)-[r:KNOWS]->(b:Person), (b)-[s:LIVES_IN]->(c:City) RETURN a, c",
        )
        .unwrap();
        assert_eq!(q.match_clauses.len(), 2);
        assert_eq!(q.match_clauses[0].elements.len(), 3);
        assert_eq!(q.match_clauses[1].elements.len(), 3);
    }

    // 7. Variable-length edge *..2
    #[test]
    fn test_variable_length_edge() {
        let q = Parser::parse("MATCH (a)-[*..2]->(b) RETURN a, b").unwrap();
        let pat = &q.match_clauses[0];
        match &pat.elements[1] {
            PatternElement::Edge(e, Direction::Out) => {
                assert!(e.var.is_none());
                assert!(e.edge_type.is_none());
                assert_eq!(e.range, Some((1, Some(2))));
            },
            other => panic!("expected edge with range, got {:?}", other),
        }
    }

    // 8. DISTINCT in RETURN
    #[test]
    fn test_distinct_return() {
        let q = Parser::parse("MATCH (n) RETURN DISTINCT n.name AS name").unwrap();
        assert_eq!(q.returns.len(), 1);
        assert!(q.returns[0].distinct);
        assert_eq!(q.returns[0].alias.as_deref(), Some("name"));
    }

    // 9. Function calls: count(*), type(r)
    #[test]
    fn test_function_calls() {
        let q = Parser::parse("MATCH (n)-[r]->(m) RETURN count(*), type(r)").unwrap();
        assert_eq!(q.returns.len(), 2);
        match &q.returns[0].expr {
            Expr::FuncCall(name, args) => {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Star));
            },
            other => panic!("expected FuncCall, got {:?}", other),
        }
        match &q.returns[1].expr {
            Expr::FuncCall(name, args) => {
                assert_eq!(name, "type");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Var(_)));
            },
            other => panic!("expected FuncCall, got {:?}", other),
        }
    }

    // 10. Incoming edge <-[r:type]-
    #[test]
    fn test_incoming_edge() {
        let q = Parser::parse("MATCH (a)<-[r:KNOWS]-(b) RETURN a, b").unwrap();
        let pat = &q.match_clauses[0];
        match &pat.elements[1] {
            PatternElement::Edge(e, Direction::In) => {
                assert_eq!(e.var.as_deref(), Some("r"));
                assert_eq!(e.edge_type.as_deref(), Some("KNOWS"));
            },
            other => panic!("expected incoming edge, got {:?}", other),
        }
    }

    // Additional: node with properties
    #[test]
    fn test_node_with_props() {
        let q = Parser::parse(r#"MATCH (n:Person {name: "Alice", age: 30}) RETURN n"#).unwrap();
        let pat = &q.match_clauses[0];
        match &pat.elements[0] {
            PatternElement::Node(np) => {
                assert_eq!(np.props.len(), 2);
                assert_eq!(np.props[0].0, "name");
                assert_eq!(np.props[0].1, Literal::String("Alice".into()));
                assert_eq!(np.props[1].0, "age");
                assert_eq!(np.props[1].1, Literal::Int(30));
            },
            _ => panic!("expected node"),
        }
    }

    // Additional: WHERE IS NULL / IS NOT NULL
    #[test]
    fn test_is_null() {
        let q = Parser::parse("MATCH (n) WHERE n.email IS NULL RETURN n").unwrap();
        assert!(matches!(&q.where_clause, Some(BoolExpr::IsNull(_))));
    }

    #[test]
    fn test_is_not_null() {
        let q = Parser::parse("MATCH (n) WHERE n.email IS NOT NULL RETURN n").unwrap();
        assert!(matches!(&q.where_clause, Some(BoolExpr::IsNotNull(_))));
    }

    // Additional: NOT in WHERE
    #[test]
    fn test_where_not() {
        let q = Parser::parse(r#"MATCH (n) WHERE NOT n.name = "Bob" RETURN n"#).unwrap();
        assert!(matches!(&q.where_clause, Some(BoolExpr::Not(_))));
    }

    // Additional: edge with no var or type (bare brackets)
    #[test]
    fn test_bare_edge() {
        let q = Parser::parse("MATCH (a)-[]->(b) RETURN a").unwrap();
        let pat = &q.match_clauses[0];
        match &pat.elements[1] {
            PatternElement::Edge(e, Direction::Out) => {
                assert!(e.var.is_none());
                assert!(e.edge_type.is_none());
                assert!(e.range.is_none());
            },
            other => panic!("expected bare edge, got {:?}", other),
        }
    }

    // Additional: variable-length with unbounded range (*)
    #[test]
    fn test_unbounded_variable_length() {
        let q = Parser::parse("MATCH (a)-[*]->(b) RETURN a").unwrap();
        let pat = &q.match_clauses[0];
        match &pat.elements[1] {
            PatternElement::Edge(e, _) => {
                assert_eq!(e.range, Some((1, None)));
            },
            other => panic!("expected edge with unbounded range, got {:?}", other),
        }
    }

    // Additional: edge with props
    #[test]
    fn test_edge_with_props() {
        let q = Parser::parse(r#"MATCH (a)-[r:KNOWS {since: 2020}]->(b) RETURN a"#).unwrap();
        let pat = &q.match_clauses[0];
        match &pat.elements[1] {
            PatternElement::Edge(e, _) => {
                assert_eq!(e.var.as_deref(), Some("r"));
                assert_eq!(e.edge_type.as_deref(), Some("KNOWS"));
                assert_eq!(e.props.len(), 1);
                assert_eq!(e.props[0].0, "since");
                assert_eq!(e.props[0].1, Literal::Int(2020));
            },
            other => panic!("expected edge with props, got {:?}", other),
        }
    }

    // Additional: CONTAINS and STARTS WITH in WHERE
    #[test]
    fn test_contains_starts_with() {
        let q = Parser::parse(r#"MATCH (n) WHERE n.name CONTAINS "ali" RETURN n"#).unwrap();
        match &q.where_clause {
            Some(BoolExpr::Comparison(_, CompOp::Contains, _)) => {},
            other => panic!("expected CONTAINS, got {:?}", other),
        }

        let q = Parser::parse(r#"MATCH (n) WHERE n.name STARTS WITH "Al" RETURN n"#).unwrap();
        match &q.where_clause {
            Some(BoolExpr::Comparison(_, CompOp::StartsWith, _)) => {},
            other => panic!("expected STARTS WITH, got {:?}", other),
        }
    }

    // Additional: multiple labels on a node
    #[test]
    fn test_multiple_labels() {
        let q = Parser::parse("MATCH (n:Person:Employee) RETURN n").unwrap();
        let pat = &q.match_clauses[0];
        match &pat.elements[0] {
            PatternElement::Node(np) => {
                assert_eq!(np.labels, vec!["Person", "Employee"]);
            },
            _ => panic!("expected node"),
        }
    }

    // Additional: parse error produces correct position
    #[test]
    fn test_parse_error() {
        let err = Parser::parse("MATCH RETURN").unwrap_err();
        match err {
            QueryError::ParseError { message, .. } => {
                assert!(message.contains("expected"));
            },
            _ => panic!("expected ParseError"),
        }
    }

    // ── DDL tests ────────────────────────────────────────────────────

    #[test]
    fn test_create_table() {
        let stmt = Parser::parse_statement(
            r#"CREATE TABLE orders (id Int64 PRIMARY KEY, customer String NOT NULL, amount Float64, node NodeRef REFERENCES GRAPH)"#,
        ).unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.name, "orders");
                assert_eq!(ct.columns.len(), 4);
                assert_eq!(ct.columns[0].name, "id");
                assert_eq!(ct.columns[0].col_type, "Int64");
                assert!(!ct.columns[0].nullable);
                assert!(ct.columns[0].is_primary_key);
                assert_eq!(ct.columns[1].name, "customer");
                assert!(!ct.columns[1].nullable);
                assert_eq!(ct.columns[2].name, "amount");
                assert!(ct.columns[2].nullable);
                assert_eq!(ct.columns[3].col_type, "NodeRef");
                // PK constraint extracted
                assert!(ct.constraints.iter().any(
                    |c| matches!(&c.kind, ConstraintKind::PrimaryKey(cols) if cols == &["id"])
                ));
            },
            other => panic!("expected CreateTable, got {:?}", other),
        }
    }

    #[test]
    fn test_drop_table() {
        let stmt = Parser::parse_statement("DROP TABLE orders").unwrap();
        match stmt {
            Statement::DropTable(name) => assert_eq!(name, "orders"),
            other => panic!("expected DropTable, got {:?}", other),
        }
    }

    // ── DML tests ────────────────────────────────────────────────────

    #[test]
    fn test_insert_positional() {
        let stmt =
            Parser::parse_statement(r#"INSERT INTO orders VALUES (1, "Alice", 99.99)"#).unwrap();
        match stmt {
            Statement::InsertInto(ins) => {
                assert_eq!(ins.table, "orders");
                assert!(ins.columns.is_none());
                assert_eq!(ins.rows.len(), 1);
                assert_eq!(ins.rows[0].len(), 3);
                assert_eq!(ins.rows[0][0], Literal::Int(1));
                assert_eq!(ins.rows[0][1], Literal::String("Alice".into()));
            },
            other => panic!("expected InsertInto, got {:?}", other),
        }
    }

    #[test]
    fn test_insert_with_columns_and_multi_row() {
        let stmt = Parser::parse_statement(
            r#"INSERT INTO orders (id, name) VALUES (1, "Alice"), (2, "Bob")"#,
        )
        .unwrap();
        match stmt {
            Statement::InsertInto(ins) => {
                assert_eq!(ins.columns, Some(vec!["id".into(), "name".into()]));
                assert_eq!(ins.rows.len(), 2);
            },
            other => panic!("expected InsertInto, got {:?}", other),
        }
    }

    #[test]
    fn test_update() {
        let stmt = Parser::parse_statement(
            r#"UPDATE orders SET status = "shipped", amount = 105.0 WHERE id = 1"#,
        )
        .unwrap();
        match stmt {
            Statement::UpdateTable(upd) => {
                assert_eq!(upd.table, "orders");
                assert_eq!(upd.assignments.len(), 2);
                assert_eq!(upd.assignments[0].0, "status");
            },
            other => panic!("expected UpdateTable, got {:?}", other),
        }
    }

    #[test]
    fn test_delete() {
        let stmt = Parser::parse_statement(r#"DELETE FROM orders WHERE id = 1"#).unwrap();
        match stmt {
            Statement::DeleteFrom(del) => {
                assert_eq!(del.table, "orders");
            },
            other => panic!("expected DeleteFrom, got {:?}", other),
        }
    }

    // ── FROM/JOIN tests ──────────────────────────────────────────────

    #[test]
    fn test_from_table_query() {
        let q = Parser::parse("FROM orders WHERE amount > 50 RETURN id, customer").unwrap();
        assert_eq!(q.from_table.as_ref().unwrap().name, "orders");
        assert!(q.from_table.as_ref().unwrap().alias.is_none());
        assert!(q.match_clauses.is_empty());
        assert_eq!(q.returns.len(), 2);
    }

    /// Regression: keyword-named columns used to fail parsing at expression
    /// position because the lexer emitted `Token::Key` and the expression
    /// parser only accepted `Token::Ident`. After the soft-keyword refactor,
    /// `key` lexes as `Token::Ident("key")` and parses everywhere an
    /// identifier is expected.
    #[test]
    fn test_keyword_named_column_qualified() {
        let q = Parser::parse(
            r#"FROM app_store WHERE app_store.key = "env_overrides" RETURN app_store.value"#,
        )
        .expect("parse");
        assert_eq!(q.from_table.as_ref().unwrap().name, "app_store");
        assert_eq!(q.returns.len(), 1);
        // RETURN expression is `Expr::Property("app_store", "value")`
        match &q.returns[0].expr {
            Expr::Property(var, prop) => {
                assert_eq!(var, "app_store");
                assert_eq!(prop, "value");
            },
            other => panic!("expected Property, got {:?}", other),
        }
        // WHERE comparison LHS is `Expr::Property("app_store", "key")`
        match q.where_clause.as_ref().unwrap() {
            BoolExpr::Comparison(lhs, _op, _rhs) => match lhs {
                Expr::Property(var, prop) => {
                    assert_eq!(var, "app_store");
                    assert_eq!(prop, "key");
                },
                other => panic!("expected Property, got {:?}", other),
            },
            other => panic!("expected Comparison, got {:?}", other),
        }
    }

    #[test]
    fn test_from_table_with_temporal() {
        let q = Parser::parse("FROM orders WHEN ALL RETURN id, amount").unwrap();
        assert_eq!(q.from_table.as_ref().unwrap().name, "orders");
        assert!(matches!(q.when, Some(WhenClause::All)));
    }

    #[test]
    fn test_table_to_table_join() {
        let q = Parser::parse(
            "FROM orders JOIN customers ON orders.customer_id = customers.id RETURN customers.name, orders.amount",
        ).unwrap();
        assert_eq!(q.from_table.as_ref().unwrap().name, "orders");
        assert_eq!(q.joins.len(), 1);
        assert_eq!(q.joins[0].table, "customers");
        match &q.joins[0].on_left {
            JoinSide::TableColumn { table, column } => {
                assert_eq!(table, "orders");
                assert_eq!(column, "customer_id");
            },
            other => panic!("expected TableColumn, got {:?}", other),
        }
    }

    #[test]
    fn test_graph_to_table_join() {
        let q = Parser::parse(
            "MATCH (n:Person) JOIN orders ON orders.node = n RETURN n.name, orders.amount",
        )
        .unwrap();
        assert_eq!(q.match_clauses.len(), 1);
        assert_eq!(q.joins.len(), 1);
        assert_eq!(q.joins[0].table, "orders");
        match &q.joins[0].on_right {
            JoinSide::GraphVar(v) => assert_eq!(v, "n"),
            other => panic!("expected GraphVar, got {:?}", other),
        }
    }

    #[test]
    fn test_multi_way_join() {
        let q = Parser::parse(
            "FROM orders JOIN customers ON orders.customer_id = customers.id JOIN shipments ON shipments.order_id = orders.id RETURN customers.name",
        ).unwrap();
        assert_eq!(q.joins.len(), 2);
        assert_eq!(q.joins[0].table, "customers");
        assert_eq!(q.joins[1].table, "shipments");
    }

    #[test]
    fn test_left_join() {
        let q = Parser::parse(
            "FROM orders LEFT JOIN returns ON orders.id = returns.order_id RETURN orders.id, returns.reason",
        ).unwrap();
        assert_eq!(q.joins.len(), 1);
        assert_eq!(q.joins[0].join_type, JoinType::Left);
        assert_eq!(q.joins[0].table, "returns");
    }

    #[test]
    fn test_table_alias() {
        let q = Parser::parse(
            "FROM orders AS o JOIN customers c ON o.customer_id = c.id RETURN o.id, c.name",
        )
        .unwrap();
        assert_eq!(q.from_table.as_ref().unwrap().name, "orders");
        assert_eq!(q.from_table.as_ref().unwrap().alias.as_deref(), Some("o"));
        assert_eq!(q.joins[0].table, "customers");
        assert_eq!(q.joins[0].alias.as_deref(), Some("c"));
    }

    #[test]
    fn test_group_by() {
        let q = Parser::parse(
            "FROM orders GROUP BY orders.status RETURN orders.status, count(orders.id) AS cnt",
        )
        .unwrap();
        assert_eq!(q.group_by.len(), 1);
        assert_eq!(q.returns.len(), 2);
    }

    #[test]
    fn test_column_default_and_autoincrement() {
        let stmt = Parser::parse_statement(
            "CREATE TABLE items (id INT AUTOINCREMENT PRIMARY KEY, name STRING NOT NULL, qty INT DEFAULT 0)",
        ).unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 3);
                assert!(ct.columns[0].autoincrement);
                assert!(ct.columns[0].is_primary_key);
                assert_eq!(ct.columns[2].default_value, Some(Literal::Int(0)));
            },
            other => panic!("expected CreateTable, got {:?}", other),
        }
    }

    #[test]
    fn test_create_index() {
        let stmt =
            Parser::parse_statement("CREATE INDEX idx_customer ON orders (customer_id)").unwrap();
        match stmt {
            Statement::CreateIndex(ci) => {
                assert_eq!(ci.index_name, "idx_customer");
                assert_eq!(ci.table, "orders");
                assert_eq!(ci.columns, vec!["customer_id"]);
                assert!(!ci.unique);
            },
            other => panic!("expected CreateIndex, got {:?}", other),
        }
    }

    #[test]
    fn test_create_unique_index() {
        let stmt =
            Parser::parse_statement("CREATE UNIQUE INDEX idx_email ON users (email)").unwrap();
        match stmt {
            Statement::CreateIndex(ci) => {
                assert!(ci.unique);
                assert_eq!(ci.columns, vec!["email"]);
            },
            other => panic!("expected CreateIndex, got {:?}", other),
        }
    }

    #[test]
    fn test_in_operator() {
        let q = Parser::parse(r#"FROM orders WHERE status IN ("pending", "shipped") RETURN id"#)
            .unwrap();
        assert!(matches!(q.where_clause, Some(BoolExpr::In(..))));
    }

    #[test]
    fn test_not_in_operator() {
        let q =
            Parser::parse(r#"FROM orders WHERE status NOT IN ("cancelled") RETURN id"#).unwrap();
        assert!(matches!(q.where_clause, Some(BoolExpr::NotIn(..))));
    }

    #[test]
    fn test_between_operator() {
        let q = Parser::parse("FROM orders WHERE amount BETWEEN 10 AND 100 RETURN id").unwrap();
        assert!(matches!(q.where_clause, Some(BoolExpr::Between(..))));
    }

    #[test]
    fn test_like_operator() {
        let q = Parser::parse(r#"FROM users WHERE name LIKE "John%" RETURN id"#).unwrap();
        assert!(matches!(q.where_clause, Some(BoolExpr::Like(..))));
    }

    #[test]
    fn test_having_clause() {
        let q = Parser::parse(
            "FROM orders GROUP BY orders.status HAVING count(orders.id) > 5 RETURN orders.status, count(orders.id) AS cnt",
        ).unwrap();
        assert_eq!(q.group_by.len(), 1);
        assert!(q.having.is_some());
    }

    #[test]
    fn test_alter_table_add_column() {
        let stmt =
            Parser::parse_statement(r#"ALTER TABLE orders ADD COLUMN notes STRING DEFAULT """#)
                .unwrap();
        match stmt {
            Statement::AlterTable(alt) => {
                assert_eq!(alt.table, "orders");
                assert_eq!(alt.add_columns.len(), 1);
                assert_eq!(alt.add_columns[0].name, "notes");
            },
            other => panic!("expected AlterTable, got {:?}", other),
        }
    }

    #[test]
    fn test_alter_table_add_multiple() {
        let stmt = Parser::parse_statement(
            "ALTER TABLE orders ADD COLUMN priority INT DEFAULT 0, ADD weight FLOAT",
        )
        .unwrap();
        match stmt {
            Statement::AlterTable(alt) => {
                assert_eq!(alt.add_columns.len(), 2);
                assert_eq!(alt.add_columns[0].name, "priority");
                assert_eq!(alt.add_columns[1].name, "weight");
            },
            other => panic!("expected AlterTable, got {:?}", other),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Soft-keyword regression matrix
    //
    // Every keyword that the soft-keyword refactor moved from specialized
    // Token to plain Ident is exercised here across every expression
    // position: bare WHERE, qualified WHERE, bare RETURN, qualified RETURN,
    // AS alias, ORDER BY, GROUP BY, inline property literal, INSERT column
    // list, UPDATE SET column, CREATE TABLE column def, and DELETE WHERE.
    // These tests lock in the post-refactor behavior — a future regression
    // that reintroduces the bug will fail exactly one of them and point
    // directly at the class of call site to fix.
    // ────────────────────────────────────────────────────────────────────

    /// Exact regression for the user-reported query that motivated the
    /// soft-keyword refactor. Parses and produces the expected AST shape.
    #[test]
    fn test_user_kv_query_qualified() {
        let q = Parser::parse(
            r#"FROM app_store WHERE app_store.key = "env_overrides" RETURN app_store.value"#,
        )
        .expect("parse");
        assert_eq!(q.from_table.as_ref().unwrap().name, "app_store");
    }

    /// Bare form: `WHERE key = "x" RETURN value` (no qualification).
    /// Phase 1+2 makes this parse; Phase 3 makes it plan and execute.
    #[test]
    fn test_user_kv_query_bare() {
        let q = Parser::parse(r#"FROM app_store WHERE key = "env_overrides" RETURN value"#)
            .expect("parse");
        assert_eq!(q.from_table.as_ref().unwrap().name, "app_store");
        assert_eq!(q.returns.len(), 1);
        match &q.returns[0].expr {
            Expr::Var(name) => assert_eq!(name, "value"),
            other => panic!("expected Var, got {:?}", other),
        }
    }

    /// Every keyword that was converted to a soft keyword, used as a
    /// qualified column reference in WHERE. The test is parameterized so
    /// the list matches the canonical soft-keyword set in the lexer.
    #[test]
    fn test_soft_keywords_in_qualified_where() {
        let soft_kws = [
            "table",
            "primary",
            "key",
            "unique",
            "references",
            "graph",
            "default",
            "autoincrement",
            "index",
            "column",
            "add",
            "into",
            "values",
            "set",
            "all",
            "last",
            "to",
        ];
        for kw in soft_kws {
            let src = format!(r#"FROM t WHERE t.{kw} = "x" RETURN t.{kw}"#, kw = kw);
            Parser::parse(&src).unwrap_or_else(|e| panic!("should parse `{}`: {:?}", kw, e));
        }
    }

    /// Same soft-keyword matrix but in bare form (unqualified). The parser
    /// produces `Expr::Var(kw)`; the planner will later promote this to
    /// `RExpr::Property(table_slot, kw)` when a single table is in scope.
    #[test]
    fn test_soft_keywords_in_bare_where() {
        let soft_kws = [
            "key", "table", "set", "graph", "all", "last", "default", "index",
        ];
        for kw in soft_kws {
            let src = format!(r#"FROM t WHERE {kw} = "x" RETURN {kw}"#, kw = kw);
            Parser::parse(&src).unwrap_or_else(|e| panic!("should parse bare `{}`: {:?}", kw, e));
        }
    }

    /// Soft keywords as AS aliases in RETURN. Previously failed because
    /// `expect_ident()` for the alias name rejected the specialized keyword
    /// tokens. Now works because the tokens are plain idents.
    #[test]
    fn test_soft_keywords_as_return_alias() {
        let q = Parser::parse(r#"FROM t RETURN t.x AS key, t.y AS table"#).unwrap();
        assert_eq!(q.returns.len(), 2);
        assert_eq!(q.returns[0].alias.as_deref(), Some("key"));
        assert_eq!(q.returns[1].alias.as_deref(), Some("table"));
    }

    /// Soft keywords as property literal keys in MATCH patterns:
    /// `MATCH (n:Person {key: "alice"})`. Previously failed at parse_props.
    #[test]
    fn test_soft_keywords_in_match_property_literal() {
        let q = Parser::parse(r#"MATCH (n:Person {key: "alice"}) RETURN n.name"#).unwrap();
        assert_eq!(q.match_clauses.len(), 1);
    }

    /// Soft keyword as column in ORDER BY (qualified) and GROUP BY.
    /// Note that MinnsQL's clause order places GROUP BY before RETURN.
    #[test]
    fn test_soft_keywords_in_order_and_group_by() {
        Parser::parse("FROM t RETURN t.name ORDER BY t.key DESC").expect("order by");
        Parser::parse("FROM t GROUP BY t.key RETURN t.key, count(*)").expect("group by");
    }

    /// The UPDATE path shares parse_or_expr with SELECT, so fixing parse_expr
    /// also fixes `UPDATE t SET v = "x" WHERE t.key = "y"`.
    #[test]
    fn test_update_with_keyword_column_in_where() {
        let stmt = Parser::parse_statement(
            r#"UPDATE app_store SET value = "new" WHERE app_store.key = "env_overrides""#,
        )
        .unwrap();
        match stmt {
            Statement::UpdateTable(upd) => {
                assert_eq!(upd.table, "app_store");
                assert_eq!(upd.assignments.len(), 1);
                assert_eq!(upd.assignments[0].0, "value");
            },
            other => panic!("expected UpdateTable, got {:?}", other),
        }
    }

    /// Same for DELETE.
    #[test]
    fn test_delete_with_keyword_column_in_where() {
        let stmt = Parser::parse_statement(
            r#"DELETE FROM app_store WHERE app_store.key = "env_overrides""#,
        )
        .unwrap();
        match stmt {
            Statement::DeleteFrom(del) => assert_eq!(del.table, "app_store"),
            other => panic!("expected DeleteFrom, got {:?}", other),
        }
    }

    /// CREATE TABLE with a column literally named `primary`. Requires the
    /// two-token lookahead in parse_create_table that distinguishes
    /// `PRIMARY KEY (...)` constraints from columns named `primary`.
    #[test]
    fn test_create_table_with_primary_column() {
        let stmt = Parser::parse_statement("CREATE TABLE t (primary INT PRIMARY KEY, name STRING)")
            .unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert_eq!(ct.columns[0].name, "primary");
                assert!(ct.columns[0].is_primary_key);
                assert_eq!(ct.columns[1].name, "name");
            },
            other => panic!("expected CreateTable, got {:?}", other),
        }
    }

    /// CREATE TABLE with a column literally named `unique`.
    #[test]
    fn test_create_table_with_unique_column() {
        let stmt =
            Parser::parse_statement("CREATE TABLE t (unique STRING NOT NULL, id INT PRIMARY KEY)")
                .unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert_eq!(ct.columns[0].name, "unique");
                assert!(!ct.columns[0].nullable);
            },
            other => panic!("expected CreateTable, got {:?}", other),
        }
    }

    /// INSERT with a soft keyword as a column name.
    #[test]
    fn test_insert_with_keyword_column() {
        let stmt = Parser::parse_statement(
            r#"INSERT INTO app_store (key, value) VALUES ("env_overrides", "data")"#,
        )
        .unwrap();
        match stmt {
            Statement::InsertInto(ins) => {
                assert_eq!(ins.table, "app_store");
                assert_eq!(
                    ins.columns.as_deref(),
                    Some(&["key".to_string(), "value".to_string()][..])
                );
            },
            other => panic!("expected InsertInto, got {:?}", other),
        }
    }

    /// Reserved operator keywords must still fail when used as identifiers
    /// in expression position. This is the negative half of the
    /// soft-keyword policy — operators stay reserved because accepting
    /// them as idents would change the parse of other queries.
    #[test]
    fn test_reserved_operators_still_fail_as_identifiers() {
        for bad in &["AND", "OR", "NOT", "IS", "IN", "BETWEEN", "LIKE"] {
            let src = format!(r#"FROM t WHERE t.{} = "x" RETURN t.{}"#, bad, bad);
            assert!(
                Parser::parse(&src).is_err(),
                "`{}` must not be usable as a column name",
                bad
            );
        }
    }

    // ── Backtick-quoted identifier escape hatch ────────────────────────

    /// Backticks let the user name a column after a genuinely reserved
    /// keyword. This is the escape hatch that makes the soft-keyword policy
    /// complete — every other column name works unquoted, and the small
    /// set of truly-reserved words is reachable via `` `name` ``.
    #[test]
    fn test_backtick_quoted_reserved_keyword_in_where() {
        let q = Parser::parse(r#"FROM t WHERE t.`where` = "x" RETURN t.`return`"#).unwrap();
        assert_eq!(q.returns.len(), 1);
        match &q.returns[0].expr {
            Expr::Property(var, prop) => {
                assert_eq!(var, "t");
                assert_eq!(prop, "return");
            },
            other => panic!("expected Property, got {:?}", other),
        }
    }

    #[test]
    fn test_backtick_quoted_reserved_keyword_in_create_table() {
        let stmt =
            Parser::parse_statement("CREATE TABLE t (`where` STRING, `select` INT PRIMARY KEY)")
                .unwrap();
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert_eq!(ct.columns[0].name, "where");
                assert_eq!(ct.columns[1].name, "select");
                assert!(ct.columns[1].is_primary_key);
            },
            other => panic!("expected CreateTable, got {:?}", other),
        }
    }
}
