use clap::Parser;
use httpmock::prelude::*;
use indoc::indoc;
use lmbench::args::Args;
use lmbench::run;

#[tokio::test]
async fn test_run_with_mock_server() {
    let server = MockServer::start();
    let hello_mock = server.mock(|when, then| {
        when.method(POST).path("/test");
        then.status(200)
            .header("content-type", "text/plain")
            .body("world");
    });

    let args = Args::parse_from(["lmbench", "-d", "hello", "-v", &server.url("/test")]);

    let stats = run(args).await.unwrap();

    hello_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.error_requests, 0);
}

#[tokio::test]
async fn test_run_with_openai_mock() {
    let server = MockServer::start();
    let openai_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"{
                    "choices": [
                        {
                            "message": {
                                "content": "hello world"
                            }
                        }
                    ]
                }"#,
            );
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "openai",
        &server.url("/v1/chat/completions"),
    ]);

    let stats = run(args).await.unwrap();

    openai_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_openai_streaming_mock() {
    let server = MockServer::start();
    let openai_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/chat/completions");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(indoc! { r#"
                data: {"choices":[{"delta":{"content":"hello"}}]}

                data: {"choices":[{"delta":{"content":" world"}}]}

                data: [DONE]

            "#});
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "openai",
        &server.url("/v1/chat/completions"),
    ]);

    let stats = run(args).await.unwrap();

    openai_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_gemini_mock() {
    let server = MockServer::start();
    let gemini_mock = server.mock(|when, then| {
        when.method(POST).path("/gemini:generateContent");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"{
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "hello world"
                                    }
                                ]
                            }
                        }
                    ]
                }"#,
            );
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "gemini",
        &server.url("/gemini:generateContent"),
    ]);

    let stats = run(args).await.unwrap();

    gemini_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_gemini_streaming_mock() {
    let server = MockServer::start();
    let gemini_mock = server.mock(|when, then| {
        when.method(POST).path("/gemini:streamGenerateContent");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"[{
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "hello world"
                                    }
                                ]
                            }
                        }
                    ]
                }]"#,
            );
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "gemini",
        &server.url("/gemini:streamGenerateContent"),
    ]);

    let stats = run(args).await.unwrap();

    gemini_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_anthropic_mock() {
    let server = MockServer::start();
    let anthropic_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/messages");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"{
                    "content": [
                        {
                            "text": "hello world"
                        }
                    ]
                }"#,
            );
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "anthropic",
        &server.url("/v1/messages"),
    ]);

    let stats = run(args).await.unwrap();

    anthropic_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_anthropic_streaming_mock() {
    let server = MockServer::start();
    let anthropic_mock = server.mock(|when, then| {
        when.method(POST).path("/v1/messages");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(indoc! { r#"
                data: {"delta":{"text":"hello"}}

                data: {"delta":{"text":" world"}}

                data: [DONE]

            "#});
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "anthropic",
        &server.url("/v1/messages"),
    ]);

    let stats = run(args).await.unwrap();

    anthropic_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_tgi_mock() {
    let server = MockServer::start();
    let tgi_mock = server.mock(|when, then| {
        when.method(POST).path("/generate");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"{
                    "generated_text": "hello world"
                }"#,
            );
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "TGI",
        &server.url("/generate"),
    ]);

    let stats = run(args).await.unwrap();

    tgi_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_tgi_streaming_mock() {
    let server = MockServer::start();
    let body = indoc! { r#"
        data: {"token":{"id":29889,"text":"Hello","logprob":-0.0028}}

        data: {"token":{"id":29889,"text":" world","logprob":-0.0022}}

    "#};
    let tgi_mock = server.mock(|when, then| {
        when.method(POST).path("/generate_stream");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(body);
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--server",
        "TGI",
        &server.url("/generate_stream"),
    ]);

    let stats = run(args).await.unwrap();

    tgi_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_custom_input_jq() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v2/models/ensemble/generate");
        then.respond_with(|req: &HttpMockRequest| {
            let echoed_body = req.body().to_string();
            HttpMockResponse::builder()
                .status(200)
                .header("content-type", "application/json")
                .body(echoed_body)
                .build()
        });
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        r#"{"question":"Hello world"}"#,
        "--tokens",
        "--input-jq",
        r#"{generated_text: [.question]}"#,
        &server.url("/v2/models/ensemble/generate"),
    ]);

    let stats = run(args).await.unwrap();

    mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_custom_output_jq() {
    let server = MockServer::start();
    let triton_mock = server.mock(|when, then| {
        when.method(POST).path("/v2/models/ensemble/generate");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"{
                  "model_name": "tensorrt_llm_bls",
                  "model_version": "1",
                  "outputs": [
                    {"name": "text_output", "data": ["Hello world"]},
                    {"name": "cum_log_probs", "data": [0.0]}
                  ]
                }"#,
            );
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--output-jq",
        r#".outputs[] | select(.name=="text_output") | .data[0]"#,
        &server.url("/v2/models/ensemble/generate"),
    ]);

    let stats = run(args).await.unwrap();

    triton_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.output_tokens, Some(2));
}

#[tokio::test]
async fn test_run_with_delay_and_duration() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/test");
        then.status(200).body("world");
    });

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--delay",
        "900",
        "--duration",
        "1",
        "--connect-timeout",
        "5",
        "-N",
        "1000",
        &server.url("/test"),
    ]);

    let stats = run(args).await.unwrap();

    assert!(mock.calls() < 1000);
    assert!(stats.success_requests < 1000);
}

#[tokio::test]
async fn test_run_with_mismatch_url() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/test");
        then.status(200).body("world");
    });

    let args = Args::parse_from(["lmbench", "-d", "hello", &server.url("/mismatch")]);

    let stats = run(args).await.unwrap();
    assert_eq!(mock.calls(), 0);
    assert_eq!(stats.success_requests, 0);
    assert_eq!(stats.error_requests, 1);
}
