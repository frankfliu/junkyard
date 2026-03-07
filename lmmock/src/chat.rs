use axum::{
    Json,
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

pub async fn chat_completions_handler(
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
        return (StatusCode::BAD_REQUEST, Json(error_body)).into_response();
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
            return (StatusCode::BAD_REQUEST, Json(error_body)).into_response();
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
                content: Some(Content::Text(
                    "Hello there! How can I help you today?".to_string(),
                )),
                reasoning_content: Some("Thinking about how to help the user...".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_string(),
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

    (StatusCode::OK, Json(response)).into_response()
}
