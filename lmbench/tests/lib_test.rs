use clap::Parser;
use httpmock::prelude::*;
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

    let args = Args::parse_from(["lmbench", "-d", "hello", "-i", &server.url("/test")]);

    let stats = run(args).await.unwrap();

    hello_mock.assert();

    assert_eq!(stats.success_requests, 1);
    assert_eq!(stats.error_requests, 0);
}
