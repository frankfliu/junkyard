use jaq_core::{Compiler, Ctx, Vars, data, load, unwrap_valr};
use jaq_json::Val;
use serde::Deserialize;
use serde_json::Value;

/// Executes a jq expression against an input JSON string using the native `jaq` library.
/// Returns a vector of all values produced by the filter.
pub fn jq(jq_expr: &str, input: &Value) -> Vec<Value> {
    let loader = load::Loader::new(
        jaq_core::defs()
            .chain(jaq_std::defs())
            .chain(jaq_json::defs()),
    );
    let arena = load::Arena::default();

    // parse the filter
    let modules = loader
        .load(
            &arena,
            load::File {
                code: jq_expr,
                path: (),
            },
        )
        .unwrap_or_else(|e| panic!("Failed to parse jq expression: {:?}", e));

    // compile the filter
    let filter = Compiler::default()
        .with_funs(
            jaq_core::funs()
                .chain(jaq_std::funs())
                .chain(jaq_json::funs()),
        )
        .compile(modules)
        .unwrap_or_else(|e| panic!("Failed to compile jq expression: {:?}", e));

    let input = Val::deserialize(input).unwrap_or_else(|e| panic!("Invalid jq input: {:?}", e));
    let ctx = Ctx::<data::JustLut<Val>>::new(&filter.lut, Vars::new([]));
    let out = filter.id.run((ctx, input)).map(unwrap_valr);

    let mut results = Vec::new();
    for res in out {
        match res
            .ok()
            .and_then(|val| serde_json::from_str(&val.to_string()).ok())
        {
            Some(val) => results.push(val),
            None => {
                return vec![];
            }
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

    #[test]
    fn test_jq_error() {
        let jq_expr = r#"{ "b": if has("b") then .b else error("Error") end }"#;
        let input_json = json!({"a": 1});
        let ret = jq(jq_expr, &input_json);
        assert!(ret.is_empty());
    }
}
