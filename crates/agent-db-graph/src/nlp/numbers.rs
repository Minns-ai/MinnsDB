//! Word-number parser: cardinals, ordinals, compounds, SI suffixes, formatted numbers.

const ONES: &[(&str, f64)] = &[
    ("zero", 0.0),
    ("one", 1.0),
    ("two", 2.0),
    ("three", 3.0),
    ("four", 4.0),
    ("five", 5.0),
    ("six", 6.0),
    ("seven", 7.0),
    ("eight", 8.0),
    ("nine", 9.0),
    ("ten", 10.0),
    ("eleven", 11.0),
    ("twelve", 12.0),
    ("thirteen", 13.0),
    ("fourteen", 14.0),
    ("fifteen", 15.0),
    ("sixteen", 16.0),
    ("seventeen", 17.0),
    ("eighteen", 18.0),
    ("nineteen", 19.0),
];

const TENS: &[(&str, f64)] = &[
    ("twenty", 20.0),
    ("thirty", 30.0),
    ("forty", 40.0),
    ("fifty", 50.0),
    ("sixty", 60.0),
    ("seventy", 70.0),
    ("eighty", 80.0),
    ("ninety", 90.0),
];

const MAGNITUDES: &[(&str, f64)] = &[
    ("hundred", 100.0),
    ("thousand", 1_000.0),
    ("million", 1_000_000.0),
    ("billion", 1_000_000_000.0),
];

const SPECIALS: &[(&str, f64)] = &[
    ("half", 0.5),
    ("quarter", 0.25),
    ("dozen", 12.0),
    ("couple", 2.0),
    ("pair", 2.0),
    ("few", 3.0),
    ("several", 5.0),
];

/// Parse a word-number expression from a slice of token strings.
/// Returns (value, tokens_consumed) if a number is found at the start.
pub fn parse_number(tokens: &[&str]) -> Option<(f64, usize)> {
    if tokens.is_empty() {
        return None;
    }

    // Try numeric token first (digits, SI suffix, comma-separated)
    if let Some(v) = parse_numeric_token(tokens[0]) {
        return Some((v, 1));
    }

    // Try "a dozen", "a couple"
    if tokens[0].eq_ignore_ascii_case("a") && tokens.len() > 1 {
        let lower = tokens[1].to_lowercase();
        for &(word, val) in SPECIALS {
            if lower == word {
                return Some((val, 2));
            }
        }
    }

    // Try specials as standalone
    {
        let lower = tokens[0].to_lowercase();
        for &(word, val) in SPECIALS {
            if lower == word {
                return Some((val, 1));
            }
        }
    }

    // Try word-number parse (greedy left-to-right)
    parse_word_number(tokens)
}

/// Parse a single token that might be numeric.
/// Handles: digits, SI suffixes (1.5K), comma-separated (1,234.56).
pub fn parse_numeric_token(s: &str) -> Option<f64> {
    if s.is_empty() {
        return None;
    }

    // Check for SI suffix at end
    let (num_part, multiplier) = if let Some(stripped) = s.strip_suffix('K').or_else(|| s.strip_suffix('k')) {
        (stripped, 1_000.0)
    } else if let Some(stripped) = s.strip_suffix('M') {
        (stripped, 1_000_000.0)
    } else if let Some(stripped) = s.strip_suffix('B') {
        (stripped, 1_000_000_000.0)
    } else if let Some(stripped) = s.strip_suffix("bn") {
        (stripped, 1_000_000_000.0)
    } else {
        (s, 1.0)
    };

    // Strip commas for parsing
    let cleaned = num_part.replace(',', "");
    cleaned.parse::<f64>().ok().map(|v| v * multiplier)
}

/// Look up a single word in the number word tables.
fn lookup_word(word: &str) -> Option<f64> {
    let lower = word.to_lowercase();
    for &(w, v) in ONES {
        if lower == w {
            return Some(v);
        }
    }
    for &(w, v) in TENS {
        if lower == w {
            return Some(v);
        }
    }
    None
}

fn is_magnitude(word: &str) -> Option<f64> {
    let lower = word.to_lowercase();
    for &(w, v) in MAGNITUDES {
        if lower == w {
            return Some(v);
        }
    }
    None
}

/// Parse compound word numbers like "twenty-seven", "one hundred fifty-three thousand".
fn parse_word_number(tokens: &[&str]) -> Option<(f64, usize)> {
    let mut total = 0.0;
    let mut current = 0.0;
    let mut consumed = 0;
    let mut found_any = false;

    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];

        // Handle hyphenated compounds like "twenty-seven"
        if token.contains('-') {
            let parts: Vec<&str> = token.split('-').collect();
            if parts.len() == 2 {
                if let (Some(tens_val), Some(ones_val)) = (lookup_word(parts[0]), lookup_word(parts[1])) {
                    current += tens_val + ones_val;
                    consumed = i + 1;
                    found_any = true;
                    i += 1;
                    continue;
                }
            }
        }

        // Try as a simple number word
        if let Some(val) = lookup_word(token) {
            current += val;
            consumed = i + 1;
            found_any = true;
            i += 1;
            continue;
        }

        // Try as magnitude
        if let Some(mag) = is_magnitude(token) {
            if !found_any {
                break;
            }
            if mag == 100.0 {
                // "two hundred" → current *= 100
                if current == 0.0 {
                    current = 1.0;
                }
                current *= mag;
            } else {
                // thousand/million/billion: accumulate group
                if current == 0.0 {
                    current = 1.0;
                }
                current *= mag;
                total += current;
                current = 0.0;
            }
            consumed = i + 1;
            i += 1;
            continue;
        }

        // "and" connector (e.g., "one hundred and fifty")
        if token.eq_ignore_ascii_case("and") && found_any && i + 1 < tokens.len() {
            // Only consume "and" if next token is also a number word
            if lookup_word(tokens[i + 1]).is_some() || tokens[i + 1].contains('-') {
                i += 1;
                continue;
            }
        }

        // Not a number word, stop
        break;
    }

    if found_any {
        total += current;
        Some((total, consumed))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_digits() {
        assert_eq!(parse_numeric_token("42"), Some(42.0));
        assert_eq!(parse_numeric_token("3.14"), Some(3.14));
    }

    #[test]
    fn test_comma_formatted() {
        assert_eq!(parse_numeric_token("1,234"), Some(1234.0));
        assert_eq!(parse_numeric_token("1,234.56"), Some(1234.56));
    }

    #[test]
    fn test_si_suffix() {
        assert_eq!(parse_numeric_token("1.5K"), Some(1500.0));
        assert_eq!(parse_numeric_token("2M"), Some(2_000_000.0));
        assert_eq!(parse_numeric_token("1B"), Some(1_000_000_000.0));
        assert_eq!(parse_numeric_token("3.5bn"), Some(3_500_000_000.0));
    }

    #[test]
    fn test_word_ones() {
        assert_eq!(parse_number(&["one"]), Some((1.0, 1)));
        assert_eq!(parse_number(&["nineteen"]), Some((19.0, 1)));
    }

    #[test]
    fn test_word_tens() {
        assert_eq!(parse_number(&["twenty"]), Some((20.0, 1)));
        assert_eq!(parse_number(&["ninety"]), Some((90.0, 1)));
    }

    #[test]
    fn test_hyphenated_compound() {
        assert_eq!(parse_number(&["twenty-seven"]), Some((27.0, 1)));
        assert_eq!(parse_number(&["fifty-three"]), Some((53.0, 1)));
    }

    #[test]
    fn test_hundred_compound() {
        assert_eq!(
            parse_number(&["one", "hundred", "fifty"]),
            Some((150.0, 3))
        );
        assert_eq!(
            parse_number(&["two", "hundred", "and", "fifty-three"]),
            Some((253.0, 4))
        );
    }

    #[test]
    fn test_thousand() {
        assert_eq!(
            parse_number(&["two", "hundred", "fifty-three", "thousand"]),
            Some((253_000.0, 4))
        );
    }

    #[test]
    fn test_specials() {
        assert_eq!(parse_number(&["half"]), Some((0.5, 1)));
        assert_eq!(parse_number(&["a", "dozen"]), Some((12.0, 2)));
        assert_eq!(parse_number(&["couple"]), Some((2.0, 1)));
    }

    #[test]
    fn test_no_number() {
        assert_eq!(parse_number(&["hello"]), None);
        assert_eq!(parse_number(&[]), None);
    }

    #[test]
    fn test_mixed_stop() {
        // "twenty euros" → only consumes "twenty"
        assert_eq!(parse_number(&["twenty", "euros"]), Some((20.0, 1)));
    }

    #[test]
    fn test_numeric_token_not_number() {
        assert_eq!(parse_numeric_token("hello"), None);
        assert_eq!(parse_numeric_token(""), None);
    }
}
