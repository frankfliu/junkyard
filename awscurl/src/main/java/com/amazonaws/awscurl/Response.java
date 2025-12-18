package com.amazonaws.awscurl;

import java.net.http.HttpHeaders;

public class Response {

    private int code;
    private String body;
    private HttpHeaders headers;

    public Response(int code, String content, HttpHeaders headers) {
        this.code = code;
        this.body = content;
        this.headers = headers;
    }

    public int statusCode() {
        return code;
    }

    public String body() {
        return body;
    }

    public HttpHeaders headers() {
        return headers;
    }
}
