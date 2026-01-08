use assert_cmd::Command;
use httpmock::prelude::*;
use predicates::str::contains;

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
