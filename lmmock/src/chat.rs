use crate::AppState;
use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Sse, sse::Event},
};
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::convert::Infallible;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrl {
    pub url: String,
    pub detail: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseFormat {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    pub store: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, i32>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub n: Option<u32>,
    pub modalities: Option<Vec<String>>,
    pub prediction: Option<Value>,
    pub audio: Option<Value>,
    pub presence_penalty: Option<f32>,
    pub response_format: Option<ResponseFormat>,
    pub seed: Option<i64>,
    pub service_tier: Option<String>,
    pub stop: Option<Value>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Logprobs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<LogprobContent>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Vec<LogprobContent>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogprobContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
    pub audio_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub audio_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

fn generate_mock_logprobs(token: &str, num_top: u32) -> Logprobs {
    Logprobs {
        content: Some(vec![LogprobContent {
            token: token.to_string(),
            logprob: -0.5,
            bytes: Some(token.as_bytes().to_vec()),
            top_logprobs: (0..num_top)
                .map(|i| TopLogprob {
                    token: format!("token_{}", i),
                    logprob: -1.0 - (i as f32),
                    bytes: None,
                })
                .collect(),
        }]),
        refusal: None,
    }
}

fn generate_mock_from_schema(schema: &Value) -> Value {
    if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
        if !enum_values.is_empty() {
            return enum_values[0].clone();
        }
    }

    match schema.get("type").and_then(|v| v.as_str()) {
        Some("string") => json!("mock_string"),
        Some("number") | Some("integer") => json!(0),
        Some("boolean") => json!(true),
        Some("object") => {
            let mut obj = serde_json::Map::new();
            if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
                for (key, sub_schema) in properties {
                    obj.insert(key.clone(), generate_mock_from_schema(sub_schema));
                }
            }
            Value::Object(obj)
        }
        Some("array") => {
            if let Some(items) = schema.get("items") {
                json!([generate_mock_from_schema(items)])
            } else {
                json!([])
            }
        }
        _ => {
            // Fallback for schemas without a "type" but with "properties"
            if schema.get("properties").is_some() {
                let mut obj = serde_json::Map::new();
                if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
                    for (key, sub_schema) in properties {
                        obj.insert(key.clone(), generate_mock_from_schema(sub_schema));
                    }
                }
                return Value::Object(obj);
            }
            json!(null)
        }
    }
}

fn to_json_response<T: Serialize>(
    status: StatusCode,
    data: T,
    pretty: bool,
) -> axum::response::Response {
    if pretty {
        let body = serde_json::to_string_pretty(&data).unwrap();
        (
            status,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            body,
        )
            .into_response()
    } else {
        (status, Json(data)).into_response()
    }
}

pub async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if payload.messages.is_empty() {
        let error_body = json!({
            "error": {
                "message": "[] is too short - 'messages'",
                "type": "invalid_request_error",
                "param": "messages",
                "code": null
            }
        });
        return to_json_response(StatusCode::BAD_REQUEST, error_body, state.pretty);
    }

    if let Some(mock_type) = headers.get("mock-result-type") {
        if mock_type == "Invalid parameter" {
            let error_body = json!({
                "error": {
                    "message": "Invalid parameter: 'model' is a required property",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "missing_required_parameter"
                }
            });
            return to_json_response(StatusCode::BAD_REQUEST, error_body, state.pretty);
        }
    }

    let logprobs_enabled = payload.logprobs.unwrap_or(false);
    let top_logprobs = payload.top_logprobs.unwrap_or(0);

    if payload.stream.unwrap_or(false) {
        let model = payload.model.clone();
        let stream = stream::unfold(0, move |state| {
            let model = model.clone();
            async move {
                match state {
                    0 => {
                        let chunk = ChatCompletionChunk {
                            id: "chatcmpl-mock".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: 1677652288,
                            model,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                            usage: None,
                            system_fingerprint: Some("fp_44706579d2".to_string()),
                        };
                        Some((
                            Ok::<_, Infallible>(Event::default().json_data(chunk).unwrap()),
                            1,
                        ))
                    }
                    1 => {
                        let content = "Hello there! ";
                        let logprobs = if logprobs_enabled {
                            Some(generate_mock_logprobs("Hello", top_logprobs))
                        } else {
                            None
                        };
                        let chunk = ChatCompletionChunk {
                            id: "chatcmpl-mock".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: 1677652288,
                            model,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChatMessageDelta {
                                    role: None,
                                    content: Some(content.to_string()),
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                logprobs,
                            }],
                            usage: None,
                            system_fingerprint: Some("fp_44706579d2".to_string()),
                        };
                        Some((
                            Ok::<_, Infallible>(Event::default().json_data(chunk).unwrap()),
                            2,
                        ))
                    }
                    2 => {
                        let content = "How can I help you today?";
                        let logprobs = if logprobs_enabled {
                            Some(generate_mock_logprobs("How", top_logprobs))
                        } else {
                            None
                        };
                        let chunk = ChatCompletionChunk {
                            id: "chatcmpl-mock".to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created: 1677652288,
                            model,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChatMessageDelta {
                                    role: None,
                                    content: Some(content.to_string()),
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: Some("stop".to_string()),
                                logprobs,
                            }],
                            usage: None,
                            system_fingerprint: Some("fp_44706579d2".to_string()),
                        };
                        Some((
                            Ok::<_, Infallible>(Event::default().json_data(chunk).unwrap()),
                            3,
                        ))
                    }
                    3 => Some((Ok::<_, Infallible>(Event::default().data("[DONE]")), 4)),
                    _ => None,
                }
            }
        });

        return Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::new())
            .into_response();
    }

    let mut metadata = HashMap::new();
    metadata.insert("weight_version".to_string(), "default".to_string());

    let logprobs = if logprobs_enabled {
        Some(generate_mock_logprobs("Hello", top_logprobs))
    } else {
        None
    };

    let mut tool_calls = None;
    let mut finish_reason = "stop".to_string();
    let mut content = Some(Content::Text(
        "Hello there! How can I help you today?".to_string(),
    ));

    if let Some(rf) = &payload.response_format {
        if rf.r#type == "json_object" {
            content = Some(Content::Text("{\n  \"message\": \"Hello!\"\n}".to_string()));
        } else if rf.r#type == "json_schema" {
            if let Some(js) = &rf.json_schema {
                if let Some(schema) = js.get("schema") {
                    let mock_val = generate_mock_from_schema(schema);
                    content = Some(Content::Text(
                        serde_json::to_string_pretty(&mock_val).unwrap(),
                    ));
                }
            }
        }
    }

    if let Some(tools) = &payload.tools {
        if !tools.is_empty() {
            let arguments = if let Some(parameters) = &tools[0].function.parameters {
                let mock_val = generate_mock_from_schema(parameters);
                serde_json::to_string(&mock_val).unwrap_or_else(|_| "{}".to_string())
            } else {
                "{}".to_string()
            };
            tool_calls = Some(vec![ToolCall {
                id: "call_abc123".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: tools[0].function.name.clone(),
                    arguments,
                },
            }]);
            finish_reason = "tool_calls".to_string();
            content = None;
        }
    }

    let response = ChatCompletionResponse {
        id: "chatcmpl-mock".to_string(),
        object: "chat.completion".to_string(),
        created: 1677652288,
        model: payload.model,
        metadata: Some(metadata),
        service_tier: Some("scale".to_string()),
        system_fingerprint: Some("fp_44706579d2".to_string()),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
                reasoning_content: Some("Thinking about how to help the user...".to_string()),
                name: None,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
            matched_stop: Some(200002),
            logprobs,
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 10,
            total_tokens: 20,
            reasoning_tokens: Some(0),
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: 0,
                audio_tokens: 0,
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: 0,
                audio_tokens: 0,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            }),
        },
    };

    to_json_response(StatusCode::OK, response, state.pretty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::post;
    use axum_test::TestServer;

    fn create_test_server() -> TestServer {
        create_test_server_with_state(AppState { pretty: false })
    }

    fn create_test_server_with_state(state: AppState) -> TestServer {
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(chat_completions_handler))
            .with_state(state);
        TestServer::new(app)
    }

    #[tokio::test]
    async fn test_pretty_output() {
        let server = create_test_server_with_state(AppState { pretty: true });
        let payload = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let text = response.text();
        // Pretty output should have newlines and indentation
        assert!(text.contains("\n  \"id\": \"chatcmpl-mock\","));
    }

    #[tokio::test]
    async fn test_chat_completions_handler_success() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let body: ChatCompletionResponse = response.json();
        assert_eq!(body.model, "gpt-4");
        assert_eq!(body.choices[0].message.role, "assistant");
        assert!(body.choices[0].logprobs.is_none());
    }

    #[tokio::test]
    async fn test_chat_completions_handler_logprobs() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": true,
            "top_logprobs": 2
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let body: ChatCompletionResponse = response.json();
        let logprobs = body.choices[0].logprobs.as_ref().unwrap();
        let content = logprobs.content.as_ref().unwrap();
        assert_eq!(content[0].top_logprobs.len(), 2);
    }

    #[tokio::test]
    async fn test_chat_completions_handler_streaming() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();

        // SSE responses are returned as text chunks.
        let text = response.text();
        assert!(text.contains("data: {\"id\":\"chatcmpl-mock\""));
        assert!(text.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_chat_completions_handler_errors() {
        let server = create_test_server();

        // Empty messages error
        let payload_empty = json!({
            "model": "gpt-4",
            "messages": []
        });
        let res_empty = server
            .post("/v1/chat/completions")
            .json(&payload_empty)
            .await;
        res_empty.assert_status(StatusCode::BAD_REQUEST);
        assert_eq!(
            res_empty.json::<Value>()["error"]["message"],
            "[] is too short - 'messages'"
        );

        // Mock invalid parameter error via header
        let payload_valid = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let res_mock = server
            .post("/v1/chat/completions")
            .add_header("mock-result-type", "Invalid parameter")
            .json(&payload_valid)
            .await;
        res_mock.assert_status(StatusCode::BAD_REQUEST);
        assert_eq!(
            res_mock.json::<Value>()["error"]["code"],
            "missing_required_parameter"
        );
    }

    #[tokio::test]
    async fn test_create_with_tools() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let body: ChatCompletionResponse = response.json();
        assert_eq!(body.object, "chat.completion");
        let choice = &body.choices[0];
        assert_eq!(choice.finish_reason, "tool_calls");
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        let arguments: Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert_eq!(arguments["location"], "mock_string");
        assert_eq!(arguments["unit"], "celsius");
    }

    #[tokio::test]
    async fn test_create_with_tools_array() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Analyze these numbers"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "numbers": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                }
                            }
                        }
                    }
                }
            ]
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let body: ChatCompletionResponse = response.json();
        let tool_calls = body.choices[0].message.tool_calls.as_ref().unwrap();
        let arguments: Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert_eq!(arguments["numbers"], json!([0]));
    }

    #[tokio::test]
    async fn test_create_with_response_format() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Output a JSON object with a 'name' field"}],
            "response_format": { "type": "json_object" }
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let body: ChatCompletionResponse = response.json();
        if let Some(Content::Text(text)) = &body.choices[0].message.content {
            let _: Value = serde_json::from_str(text).expect("Response should be valid JSON");
            assert!(text.contains("\"message\": \"Hello!\""));
        } else {
            panic!("Expected text content");
        }
    }

    #[tokio::test]
    async fn test_create_with_structured_outputs() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": { "type": "string" }
                        },
                        "required": ["answer"],
                        "additionalProperties": false
                    },
                    "strict": true
                }
            }
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let body: ChatCompletionResponse = response.json();
        if let Some(Content::Text(text)) = &body.choices[0].message.content {
            let val: Value = serde_json::from_str(text).expect("Response should be valid JSON");
            assert_eq!(val["answer"], "mock_string");
        } else {
            panic!("Expected text content");
        }
    }

    #[tokio::test]
    async fn test_create_with_content_parts() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gnome-home.svg/1200px-Gnome-home.svg.png",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
    }

    #[tokio::test]
    async fn test_create_with_all_params() {
        let server = create_test_server();
        let payload = json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "max_completion_tokens": 100,
            "n": 1,
            "seed": 42,
            "stop": ["\n", "User:"],
            "temperature": 0.8,
            "top_p": 1.0,
            "user": "test-user"
        });

        let response = server.post("/v1/chat/completions").json(&payload).await;
        response.assert_status_ok();
        let body: ChatCompletionResponse = response.json();
        assert_eq!(body.usage.total_tokens, 20);
    }
}
