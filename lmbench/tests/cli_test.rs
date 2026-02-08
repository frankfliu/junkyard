use assert_cmd::Command;
use httpmock::prelude::*;
use predicates::str::contains;
use tempfile::tempdir;

#[tokio::test]
async fn test_main_with_mock_server() {
    let server = MockServer::start();
    let hello_mock = server.mock(|when, then| {
        when.method(POST).path("/test");
        then.status(200)
            .header("content-type", "text/plain")
            .body("world");
    });

    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("lmbench"));
    cmd.arg(server.url("/test"))
        .arg("-d")
        .arg("hello")
        .arg("-v");

    cmd.assert().success().stdout(contains("Status: 200 OK"));

    hello_mock.assert();
}

#[tokio::test]
async fn test_main_with_output() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/test");
        then.status(200).body("world");
    });

    let dir = tempdir().unwrap();
    let output_path = dir.path().to_str().unwrap();

    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("lmbench"));
    cmd.arg(server.url("/test"))
        .arg("-d")
        .arg("hello")
        .arg("--repeat")
        .arg("2")
        .arg("-c")
        .arg("2")
        .arg("--output")
        .arg(output_path)
        .env("RUST_LOG", "info");

    cmd.assert().success();

    assert_eq!(mock.calls(), 4);

    let content = std::fs::read_to_string(dir.path().join("output.log")).unwrap();
    assert_eq!(content.lines().count(), 4);
    assert!(content.contains(r#""generated_text":["world"]"#));

    // with count token
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("lmbench"));
    cmd.arg(server.url("/test"))
        .arg("-d")
        .arg("hello")
        .arg("--tokens")
        .arg("--output")
        .arg(output_path);

    cmd.assert().success();

    let content = std::fs::read_to_string(dir.path().join("output.log")).unwrap();
    assert!(content.contains(r#""generated_text":["world"]"#));
    assert!(content.contains(r#""benchmark_output_tokens":1"#));

    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("lmbench"));
    cmd.arg(server.url("/test"))
        .arg("-d")
        .arg("hello")
        .arg("--tokens")
        .arg("-v")
        .arg("--output")
        .arg(output_path);

    cmd.assert().success();

    let content = std::fs::read_to_string(dir.path().join("output.log")).unwrap();
    assert!(content.contains("headers"));
    assert!(content.contains(r#""benchmark_output_tokens":1"#));
}
