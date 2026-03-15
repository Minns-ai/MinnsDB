use super::ast::*;
use super::lexer::Lexer;
use super::token::{Spanned, Token};
use super::types::QueryError;

pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    pub fn parse(input: &str) -> Result<Query, QueryError> {
        let tokens = Lexer::tokenize(input).map_err(|e| QueryError::ParseError {
            message: e.message,
            position: e.position,
        })?;
        let mut parser = Parser { tokens, pos: 0 };
        parser.parse_query()
    }

    // ── Top-level ───────────────────────────────────────────────────────

    fn parse_query(&mut self) -> Result<Query, QueryError> {
        let match_clauses = self.parse_match()?;
        let when = self.parse_when()?;
        let as_of = self.parse_as_of()?;
        let where_clause = self.parse_where()?;
        let returns = self.parse_return()?;
        let order_by = self.parse_order_by()?;
        let limit = self.parse_limit()?;

        if !self.at(&Token::Eof) {
            return Err(self.error(format!("unexpected token {:?}", self.peek())));
        }

        Ok(Query {
            match_clauses,
            when,
            as_of,
            where_clause,
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
        if self.at(&Token::All) {
            self.advance();
            return Ok(Some(WhenClause::All));
        }

        // WHEN LAST "duration"
        if self.at(&Token::Last) {
            self.advance();
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
        if self.at(&Token::To) {
            self.advance();
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

    fn parse_or_expr(&mut self) -> Result<BoolExpr, QueryError> {
        let mut left = self.parse_and_expr()?;
        while self.at(&Token::Or) {
            self.advance();
            let right = self.parse_and_expr()?;
            left = BoolExpr::Or(Box::new(left), Box::new(right));
        }
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
            let inner = self.parse_not_expr()?;
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
            _ => Err(self.error(format!("expected literal, found {:?}", self.peek()))),
        }
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
}
