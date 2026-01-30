use jaq_interpret::{Ctx, FilterT, ParseCtx, RcIter, Val};
use serde_json::Value;

/// Executes a jq expression against an input JSON string using the native `jaq` library.
/// Returns a vector of all values produced by the filter.
pub fn jq(jq_expr: &str, input: &Value) -> Vec<Value> {
    let mut pctx = ParseCtx::new(Vec::new());
    pctx.insert_natives(jaq_core::core());
    pctx.insert_defs(jaq_std::std());

    let (f, errs) = jaq_parse::parse(jq_expr, jaq_parse::main());
    if !errs.is_empty() {
        panic!("Failed to parse jq expression: {:?}", errs);
    }

    let f = pctx.compile(f.expect("Parsed filter is None"));

    if !pctx.errs.is_empty() {
        panic!(
            "Failed to compile jq expression: {} errors occurred",
            pctx.errs.len()
        );
    }

    let inputs = RcIter::new(core::iter::empty());
    let out = f.run((Ctx::new([], &inputs), Val::from(input.clone())));

    let mut results = Vec::new();
    for res in out {
        if let Ok(val) = res {
            results.push(Value::from(val));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_jq_object_construction() {
        let jq_expr = r#"{user_id: .user_id, "detail": { "level": .shop_level}}"#;
        let input_json = json!({ "user_id": 101,  "shop_level": "Admin"});
        let expected = vec![json!({
            "user_id": 101,
            "detail": {
                "level": "Admin"
            }
        })];
        assert_eq!(jq(jq_expr, &input_json), expected);
    }

    #[test]
    fn test_jq_simple_field() {
        let jq_expr = ".user_id";
        let input_json = json!({ "user_id": 101,  "shop_level": "Admin"});
        let expected = vec![json!(101)];
        assert_eq!(jq(jq_expr, &input_json), expected);
    }

    #[test]
    fn test_jq_select() {
        let jq_expr = "select(.user_id == 101)";
        let input_json = json!({ "user_id": 101,  "shop_level": "Admin"});
        let expected = vec![json!({ "user_id": 101,  "shop_level": "Admin"})];
        assert_eq!(jq(jq_expr, &input_json), expected);
    }

    #[test]
    fn test_jq_multiple_results() {
        let jq_expr = ".[]";
        let input_json = json!([1, 2, 3]);
        let expected = vec![json!(1), json!(2), json!(3)];
        assert_eq!(jq(jq_expr, &input_json), expected);
    }
}
