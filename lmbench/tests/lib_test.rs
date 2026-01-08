use clap::Parser;
use httpmock::prelude::*;
use indoc::indoc;
use lmbench::args::Args;
use lmbench::run;
use tempfile::tempdir;

#[tokio::test]
async fn test_run_with_output() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/test");
        then.status(200).body("world");
    });

    let dir = tempdir().unwrap();
    let output_path = dir.path().to_str().unwrap();

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--repeat",
        "2",
        "-c",
        "2",
        "--output",
        output_path,
        &server.url("/test"),
    ]);

    let stats = run(args).await.unwrap();

    assert_eq!(mock.hits(), 4);
    assert_eq!(stats.success_requests, 4);

    let content = std::fs::read_to_string(dir.path().join("data.jsonlines")).unwrap();
    assert_eq!(content.lines().count(), 4);
    assert!(content.contains(r#""response":"world"#));

    // with count token
    let args = Args::parse_from([
        "lmbench",
        "-d",
        "hello",
        "--tokens",
        "--output",
        output_path,
        &server.url("/test"),
    ]);

    let stats = run(args).await.unwrap();
    assert_eq!(stats.success_requests, 1);

    let content = std::fs::read_to_string(dir.path().join("data.jsonlines")).unwrap();
    assert!(content.contains(r#""generated_text":"world""#));
    assert!(content.contains(r#""token_count":1"#));

    let args = Args::parse_from([
        "lmbench",
        "-d",
        "data",
        "-v",
        "--tokens",
        "--output",
        output_path,
        &server.url("/test"),
    ]);

    let stats = run(args).await.unwrap();
    assert_eq!(stats.success_requests, 1);

    let content = std::fs::read_to_string(dir.path().join("data.jsonlines")).unwrap();
    assert!(content.contains("headers"));
    assert!(content.contains(r#""token_count":1"#));
}

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
        "-j",
        "openai",
        &server.url("/v1/chat/completions"),
    ]);

    let stats = run(args).await.unwrap();

    openai_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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
        "-j",
        "openai",
        &server.url("/v1/chat/completions"),
    ]);

    let stats = run(args).await.unwrap();

    openai_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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
        "-j",
        "gemini",
        &server.url("/gemini:generateContent"),
    ]);

    let stats = run(args).await.unwrap();

    gemini_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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
        "-j",
        "gemini",
        &server.url("/gemini:streamGenerateContent"),
    ]);

    let stats = run(args).await.unwrap();

    gemini_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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
        "-j",
        "anthropic",
        &server.url("/v1/messages"),
    ]);

    let stats = run(args).await.unwrap();

    anthropic_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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
        "-j",
        "anthropic",
        &server.url("/v1/messages"),
    ]);

    let stats = run(args).await.unwrap();

    anthropic_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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
        "-j",
        "TGI",
        &server.url("/generate"),
    ]);

    let stats = run(args).await.unwrap();

    tgi_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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
        "-j",
        "TGI",
        &server.url("/generate_stream"),
    ]);

    let stats = run(args).await.unwrap();

    tgi_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
}

#[tokio::test]
async fn test_run_with_custom_jq() {
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
        "-j",
        r#"$.outputs[?(@.name=="text_output")].data[0]"#,
        &server.url("/v2/models/ensemble/generate"),
    ]);

    let stats = run(args).await.unwrap();

    triton_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.total_output_tokens, 2);
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

    assert!(mock.hits() < 1000);
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
    assert_eq!(mock.hits(), 0);
    assert_eq!(stats.success_requests, 0);
    assert_eq!(stats.error_requests, 1);
}
