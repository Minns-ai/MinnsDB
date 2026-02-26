//! Domain gazetteers: currency, relations, places, categories, split signals.

/// Match result from a gazetteer lookup.
#[derive(Debug, Clone)]
pub struct GazetteerMatch {
    pub text: String,
    pub category: GazetteerCategory,
    pub confidence: f32,
}

/// Categories of gazetteer entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GazetteerCategory {
    Currency,
    Relation,
    SentimentPositive,
    SentimentNegative,
    CategoryArt,
    CategoryMusic,
    CategoryFood,
    CategorySport,
    CategoryFilm,
    CategoryBooks,
    SplitSignal,
    TemporalCue,
}

// ---- Currency gazetteer ----
const CURRENCIES: &[(&str, &str)] = &[
    ("$", "USD"),
    ("aud", "AUD"),
    ("baht", "THB"),
    ("brl", "BRL"),
    ("bucks", "USD"),
    ("cad", "CAD"),
    ("cent", "USD"),
    ("cents", "USD"),
    ("chf", "CHF"),
    ("cny", "CNY"),
    ("czk", "CZK"),
    ("dkk", "DKK"),
    ("dollar", "USD"),
    ("dollars", "USD"),
    ("euro", "EUR"),
    ("euros", "EUR"),
    ("eur", "EUR"),
    ("forint", "HUF"),
    ("franc", "CHF"),
    ("francs", "CHF"),
    ("gbp", "GBP"),
    ("hkd", "HKD"),
    ("huf", "HUF"),
    ("inr", "INR"),
    ("koruna", "CZK"),
    ("krone", "NOK"),
    ("krona", "SEK"),
    ("krw", "KRW"),
    ("lira", "TRY"),
    ("mxn", "MXN"),
    ("nok", "NOK"),
    ("nzd", "NZD"),
    ("pence", "GBP"),
    ("penny", "GBP"),
    ("peso", "MXN"),
    ("pesos", "MXN"),
    ("pln", "PLN"),
    ("pound", "GBP"),
    ("pounds", "GBP"),
    ("quid", "GBP"),
    ("rand", "ZAR"),
    ("real", "BRL"),
    ("renminbi", "CNY"),
    ("rub", "RUB"),
    ("ruble", "RUB"),
    ("rubles", "RUB"),
    ("rupee", "INR"),
    ("rupees", "INR"),
    ("sek", "SEK"),
    ("sgd", "SGD"),
    ("thb", "THB"),
    ("try", "TRY"),
    ("twd", "TWD"),
    ("usd", "USD"),
    ("won", "KRW"),
    ("yen", "JPY"),
    ("yuan", "CNY"),
    ("zar", "ZAR"),
    ("zloty", "PLN"),
    ("£", "GBP"),
    ("¥", "JPY"),
    ("€", "EUR"),
    ("₦", "NGN"),
    ("₩", "KRW"),
    ("₫", "VND"),
    ("₴", "UAH"),
    ("₹", "INR"),
    ("₺", "TRY"),
    ("₽", "RUB"),
    ("฿", "THB"),
];

// ---- Relation types ----
const RELATIONS: &[(&str, &str)] = &[
    ("acquaintance", "acquaintance"),
    ("boss", "boss"),
    ("brother", "sibling"),
    ("child", "child"),
    ("classmate", "classmate"),
    ("client", "client"),
    ("colleague", "colleague"),
    ("cousin", "cousin"),
    ("coworker", "colleague"),
    ("friend", "friend"),
    ("husband", "spouse"),
    ("manager", "manager"),
    ("mentor", "mentor"),
    ("neighbor", "neighbor"),
    ("neighbour", "neighbor"),
    ("parent", "parent"),
    ("partner", "partner"),
    ("roommate", "roommate"),
    ("sibling", "sibling"),
    ("sister", "sibling"),
    ("spouse", "spouse"),
    ("student", "student"),
    ("supervisor", "supervisor"),
    ("teacher", "teacher"),
    ("teammate", "teammate"),
    ("wife", "spouse"),
];

// ---- Split signals ----
const SPLIT_SIGNALS: &[&str] = &[
    "among",
    "apiece",
    "altogether",
    "between",
    "divided",
    "each",
    "equally",
    "everyone",
    "per person",
    "shared",
    "split",
];

// ---- Temporal cues ----
const TEMPORAL_CUES: &[&str] = &[
    "afternoon",
    "always",
    "daily",
    "evening",
    "every",
    "friday",
    "monday",
    "monthly",
    "morning",
    "night",
    "saturday",
    "sometimes",
    "sunday",
    "thursday",
    "tuesday",
    "usually",
    "wednesday",
    "weekly",
];

// ---- Category gazetteers ----
const ART_WORDS: &[&str] = &[
    "abstract",
    "acrylic",
    "art",
    "artistic",
    "baroque",
    "canvas",
    "ceramics",
    "charcoal",
    "classical",
    "collage",
    "contemporary",
    "cubism",
    "dali",
    "da vinci",
    "drawing",
    "etching",
    "exhibit",
    "exhibition",
    "expressionism",
    "fresco",
    "gallery",
    "gothic",
    "illustration",
    "impressionism",
    "installation",
    "landscape",
    "lithograph",
    "matisse",
    "medieval",
    "michelangelo",
    "modern art",
    "monet",
    "mosaic",
    "mural",
    "museum",
    "oil painting",
    "painting",
    "palette",
    "pastel",
    "photography",
    "picasso",
    "portrait",
    "pottery",
    "print",
    "realism",
    "rembrandt",
    "renaissance",
    "rodin",
    "sculpture",
    "sketch",
    "statue",
    "studio",
    "surrealism",
    "tapestry",
    "van gogh",
    "watercolor",
    "woodcut",
];

const MUSIC_WORDS: &[&str] = &[
    "album",
    "band",
    "bass",
    "beethoven",
    "blues",
    "brass",
    "cello",
    "choir",
    "chopin",
    "chord",
    "classical",
    "composer",
    "concert",
    "conductor",
    "drum",
    "drums",
    "ensemble",
    "flute",
    "folk",
    "guitar",
    "harmony",
    "hip-hop",
    "instrument",
    "jazz",
    "lyrics",
    "melody",
    "mozart",
    "musician",
    "note",
    "opera",
    "orchestra",
    "organ",
    "performer",
    "piano",
    "playlist",
    "pop",
    "rap",
    "record",
    "rhythm",
    "rock",
    "saxophone",
    "singer",
    "solo",
    "song",
    "soprano",
    "soundtrack",
    "strings",
    "symphony",
    "tenor",
    "track",
    "trumpet",
    "tune",
    "viola",
    "violin",
    "vocal",
];

const FOOD_WORDS: &[&str] = &[
    "appetizer",
    "bakery",
    "barbecue",
    "bistro",
    "brunch",
    "buffet",
    "burger",
    "cafe",
    "cafeteria",
    "cake",
    "catering",
    "chef",
    "chocolate",
    "cocktail",
    "coffee",
    "cook",
    "cooking",
    "course",
    "cuisine",
    "curry",
    "deli",
    "dessert",
    "diner",
    "dining",
    "dinner",
    "dish",
    "espresso",
    "feast",
    "food",
    "gourmet",
    "grill",
    "ice cream",
    "ingredient",
    "kitchen",
    "lunch",
    "meal",
    "menu",
    "michelin",
    "noodle",
    "organic",
    "pasta",
    "pastry",
    "pizza",
    "recipe",
    "restaurant",
    "salad",
    "sandwich",
    "sauce",
    "seafood",
    "snack",
    "soup",
    "spice",
    "steak",
    "sushi",
    "tapas",
    "tea",
    "vegan",
    "vegetarian",
    "wine",
];

const SPORT_WORDS: &[&str] = &[
    "athlete",
    "athletics",
    "badminton",
    "baseball",
    "basketball",
    "boxing",
    "championship",
    "climb",
    "climbing",
    "coach",
    "compete",
    "competition",
    "cricket",
    "cycling",
    "diving",
    "exercise",
    "fencing",
    "fitness",
    "football",
    "golf",
    "gym",
    "gymnastics",
    "handball",
    "hiking",
    "hockey",
    "judo",
    "karate",
    "league",
    "marathon",
    "match",
    "olympics",
    "player",
    "polo",
    "race",
    "rowing",
    "rugby",
    "running",
    "sailing",
    "ski",
    "skiing",
    "soccer",
    "sport",
    "sports",
    "sprint",
    "squash",
    "stadium",
    "surfing",
    "swim",
    "swimming",
    "team",
    "tennis",
    "tournament",
    "track",
    "training",
    "triathlon",
    "volleyball",
    "wrestling",
    "yoga",
];

const FILM_WORDS: &[&str] = &[
    "actor",
    "actress",
    "animation",
    "blockbuster",
    "box office",
    "camera",
    "cast",
    "character",
    "cinema",
    "cinematography",
    "comedy",
    "director",
    "documentary",
    "drama",
    "film",
    "filming",
    "genre",
    "horror",
    "imax",
    "movie",
    "movies",
    "oscar",
    "performance",
    "plot",
    "premiere",
    "producer",
    "production",
    "role",
    "scene",
    "screen",
    "screenplay",
    "screening",
    "script",
    "sequel",
    "series",
    "short film",
    "show",
    "special effects",
    "studio",
    "thriller",
    "trailer",
];

const BOOK_WORDS: &[&str] = &[
    "article",
    "author",
    "autobiography",
    "bestseller",
    "biography",
    "blog",
    "book",
    "bookshop",
    "chapter",
    "classic",
    "edition",
    "essay",
    "fairy tale",
    "fantasy",
    "fiction",
    "hardcover",
    "journal",
    "library",
    "literary",
    "literature",
    "memoir",
    "mystery",
    "narrative",
    "non-fiction",
    "novel",
    "novelist",
    "page",
    "paperback",
    "poem",
    "poet",
    "poetry",
    "prose",
    "publication",
    "publisher",
    "read",
    "reader",
    "reading",
    "romance",
    "sci-fi",
    "science fiction",
    "story",
    "tale",
    "textbook",
    "thriller",
    "volume",
    "writer",
    "writing",
];

/// Check if a word/phrase matches any gazetteer. Returns all matches.
pub fn lookup(text: &str) -> Vec<GazetteerMatch> {
    let lower = text.to_lowercase();
    let mut matches = Vec::new();

    if is_currency(&lower).is_some() {
        matches.push(GazetteerMatch {
            text: text.to_string(),
            category: GazetteerCategory::Currency,
            confidence: 1.0,
        });
    }

    if is_relation(&lower).is_some() {
        matches.push(GazetteerMatch {
            text: text.to_string(),
            category: GazetteerCategory::Relation,
            confidence: 1.0,
        });
    }

    if is_split_signal(&lower) {
        matches.push(GazetteerMatch {
            text: text.to_string(),
            category: GazetteerCategory::SplitSignal,
            confidence: 0.9,
        });
    }

    if is_temporal_cue(&lower) {
        matches.push(GazetteerMatch {
            text: text.to_string(),
            category: GazetteerCategory::TemporalCue,
            confidence: 0.8,
        });
    }

    if let Some(cat) = classify_category(&lower) {
        matches.push(GazetteerMatch {
            text: text.to_string(),
            category: cat,
            confidence: 0.85,
        });
    }

    matches
}

/// Check if a word is a known currency name/code/symbol. Returns canonical code.
pub fn is_currency(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    for &(name, code) in CURRENCIES {
        if lower == name {
            return Some(code);
        }
    }
    None
}

/// Check if a word is a relation type. Returns canonical type.
pub fn is_relation(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    for &(name, canonical) in RELATIONS {
        if lower == name {
            return Some(canonical);
        }
    }
    None
}

/// Check if a word is a split signal.
pub fn is_split_signal(text: &str) -> bool {
    let lower = text.to_lowercase();
    SPLIT_SIGNALS.iter().any(|&s| lower == s)
}

/// Check if a word is a temporal cue.
pub fn is_temporal_cue(text: &str) -> bool {
    let lower = text.to_lowercase();
    TEMPORAL_CUES.iter().any(|&s| lower == s)
}

/// Classify a word into a content category (art, music, food, etc.).
pub fn classify_category(text: &str) -> Option<GazetteerCategory> {
    let lower = text.to_lowercase();
    if ART_WORDS.iter().any(|&w| lower == w) {
        return Some(GazetteerCategory::CategoryArt);
    }
    if MUSIC_WORDS.iter().any(|&w| lower == w) {
        return Some(GazetteerCategory::CategoryMusic);
    }
    if FOOD_WORDS.iter().any(|&w| lower == w) {
        return Some(GazetteerCategory::CategoryFood);
    }
    if SPORT_WORDS.iter().any(|&w| lower == w) {
        return Some(GazetteerCategory::CategorySport);
    }
    if FILM_WORDS.iter().any(|&w| lower == w) {
        return Some(GazetteerCategory::CategoryFilm);
    }
    if BOOK_WORDS.iter().any(|&w| lower == w) {
        return Some(GazetteerCategory::CategoryBooks);
    }
    None
}

/// Map a gazetteer category to a preference category string.
pub fn category_to_preference(cat: GazetteerCategory) -> Option<&'static str> {
    match cat {
        GazetteerCategory::CategoryArt => Some("art"),
        GazetteerCategory::CategoryMusic => Some("music"),
        GazetteerCategory::CategoryFood => Some("food"),
        GazetteerCategory::CategorySport => Some("sports"),
        GazetteerCategory::CategoryFilm => Some("movies"),
        GazetteerCategory::CategoryBooks => Some("books"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_currency_lookup() {
        assert_eq!(is_currency("EUR"), Some("EUR"));
        assert_eq!(is_currency("euro"), Some("EUR"));
        assert_eq!(is_currency("euros"), Some("EUR"));
        assert_eq!(is_currency("€"), Some("EUR"));
        assert_eq!(is_currency("dollar"), Some("USD"));
        assert_eq!(is_currency("$"), Some("USD"));
        assert_eq!(is_currency("quid"), Some("GBP"));
        assert_eq!(is_currency("bucks"), Some("USD"));
        assert_eq!(is_currency("xyz"), None);
    }

    #[test]
    fn test_relation_lookup() {
        assert_eq!(is_relation("colleague"), Some("colleague"));
        assert_eq!(is_relation("coworker"), Some("colleague"));
        assert_eq!(is_relation("friend"), Some("friend"));
        assert_eq!(is_relation("wife"), Some("spouse"));
        assert_eq!(is_relation("neighbour"), Some("neighbor"));
        assert_eq!(is_relation("xyz"), None);
    }

    #[test]
    fn test_split_signal() {
        assert!(is_split_signal("split"));
        assert!(is_split_signal("equally"));
        assert!(is_split_signal("each"));
        assert!(!is_split_signal("hello"));
    }

    #[test]
    fn test_category_classification() {
        assert_eq!(
            classify_category("museum"),
            Some(GazetteerCategory::CategoryArt)
        );
        assert_eq!(
            classify_category("monet"),
            Some(GazetteerCategory::CategoryArt)
        );
        assert_eq!(
            classify_category("jazz"),
            Some(GazetteerCategory::CategoryMusic)
        );
        assert_eq!(
            classify_category("restaurant"),
            Some(GazetteerCategory::CategoryFood)
        );
        assert_eq!(
            classify_category("tennis"),
            Some(GazetteerCategory::CategorySport)
        );
        assert_eq!(
            classify_category("cinema"),
            Some(GazetteerCategory::CategoryFilm)
        );
        assert_eq!(
            classify_category("novel"),
            Some(GazetteerCategory::CategoryBooks)
        );
        assert_eq!(classify_category("xyz"), None);
    }

    #[test]
    fn test_multi_match() {
        let matches = lookup("morning");
        assert!(matches
            .iter()
            .any(|m| m.category == GazetteerCategory::TemporalCue));
    }
}
