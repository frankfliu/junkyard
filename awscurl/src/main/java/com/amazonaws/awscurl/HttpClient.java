package com.amazonaws.awscurl;

import com.google.gson.JsonElement;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.NullOutputStream;
import org.apache.http.Header;
import org.apache.http.HttpResponse;
import org.apache.http.HttpVersion;
import org.apache.http.StatusLine;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpDelete;
import org.apache.http.client.methods.HttpEntityEnclosingRequestBase;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.client.methods.HttpUriRequest;
import org.apache.http.conn.ssl.NoopHostnameVerifier;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.conn.ssl.TrustAllStrategy;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.message.BasicHttpResponse;
import org.apache.http.message.BasicStatusLine;
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;

@SuppressWarnings("PMD.SystemPrintln")
public final class HttpClient {

    private HttpClient() {}

    public static HttpResponse sendRequest(
            SignableRequest request,
            boolean insecure,
            int timeout,
            OutputStream ps,
            boolean dumpHeader,
            AtomicInteger tokens,
            long[] firstToken,
            String jsonExpression)
            throws IOException {
        try (CloseableHttpClient client = getHttpClient(insecure, timeout)) {
            HttpUriRequest req =
                    createHttpRequest(
                            request.getHttpMethod(), request.getUri(), request.getContent());
            if (dumpHeader) {
                String path = request.getUri().getPath();
                if (!path.startsWith("/") && !path.isEmpty()) {
                    path = path.substring(1);
                }
                System.out.println("> " + request.getHttpMethod() + " /" + path + " HTTP/1.1");
                System.out.println("> ");
            }

            addHeaders(req, request.getHeaders(), dumpHeader);
            addHeaders(req, request.getSignedHeaders(), dumpHeader);

            HttpResponse resp = client.execute(req);
            int code = resp.getStatusLine().getStatusCode();
            if (dumpHeader) {
                System.out.println("> ");
                System.out.println(
                        "< HTTP/1.1 " + code + ' ' + resp.getStatusLine().getReasonPhrase());
                System.out.println("< ");
                for (Header header : resp.getAllHeaders()) {
                    System.out.println("< " + header.getName() + ": " + header.getValue());
                }
                System.out.println("< ");
            }

            if (code >= 300 && ps instanceof NullOutputStream) {
                System.out.println(
                        "HTTP error ("
                                + resp.getStatusLine()
                                + "): "
                                + IOUtils.toString(
                                        resp.getEntity().getContent(), StandardCharsets.UTF_8));
                return resp;
            }

            Header header = resp.getFirstHeader("Content-Type");
            String contentType = header == null ? null : header.getValue();
            if (tokens != null) {
                if (contentType == null || "text/plain".equals(contentType)) {
                    String body = EntityUtils.toString(resp.getEntity());
                    ps.write(body.getBytes(StandardCharsets.UTF_8));
                    updateTokenCount(Collections.singletonList(body), tokens, request);
                    return resp;
                } else if ("application/json".equals(contentType)) {
                    String body = EntityUtils.toString(resp.getEntity());
                    ps.write(body.getBytes(StandardCharsets.UTF_8));

                    JsonElement element = JsonUtils.GSON.fromJson(body, JsonElement.class);
                    List<String> lines = new ArrayList<>();
                    JsonUtils.getJsonList(element, lines, jsonExpression);
                    updateTokenCount(lines, tokens, request);
                    return resp;
                } else if ("application/jsonlines".equals(contentType)) {
                    InputStream is = resp.getEntity().getContent();
                    boolean hasError = false;
                    try (BufferedReader reader =
                            new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                        String line;
                        List<StringBuilder> list = new ArrayList<>();
                        while ((line = reader.readLine()) != null) {
                            hasError =
                                    JsonUtils.processJsonLine(
                                                    list, firstToken, ps, line, jsonExpression)
                                            || hasError;
                        }
                        updateTokenCount(list, tokens, request);
                    }
                    if (hasError) {
                        StatusLine status = new BasicStatusLine(HttpVersion.HTTP_1_1, 500, "error");
                        return new BasicHttpResponse(status);
                    }
                    return resp;
                } else if ("application/vnd.amazon.eventstream".equals(contentType)) {
                    List<StringBuilder> list = new ArrayList<>();
                    InputStream is = resp.getEntity().getContent();
                    handleEventStream(is, list, firstToken, jsonExpression, ps);
                    updateTokenCount(list, tokens, request);
                    return resp;
                }
            }

            try (InputStream is = resp.getEntity().getContent()) {
                if ("application/vnd.amazon.eventstream".equals(contentType)) {
                    List<StringBuilder> list = new ArrayList<>();
                    handleEventStream(is, list, firstToken, jsonExpression, ps);
                } else {
                    IOUtils.copy(is, ps);
                    ps.flush();
                }
            }
            return resp;
        }
    }

    private static void handleEventStream(
            InputStream is,
            List<StringBuilder> list,
            long[] firstToken,
            String jsonExpression,
            OutputStream ps)
            throws IOException {
        byte[] buf = new byte[12];
        byte[] payload = new byte[512];
        while (true) {
            try {
                IOUtils.readFully(is, buf);
                ByteBuffer bb = ByteBuffer.wrap(buf);
                bb.order(ByteOrder.BIG_ENDIAN);
                int totalLength = bb.getInt();
                int headerLength = bb.getInt();
                int payloadLength = totalLength - headerLength - 12 - 4;
                int size = totalLength - 12;
                if (size > payload.length) {
                    payload = new byte[size];
                }
                IOUtils.readFully(is, payload, 0, size);
                if (payloadLength == 0) {
                    break;
                }
                String line =
                        new String(payload, headerLength, payloadLength, StandardCharsets.UTF_8)
                                .trim();
                if (JsonUtils.processJsonLine(list, firstToken, ps, line, jsonExpression)) {
                    throw new IOException("Response contains error");
                }
            } catch (EOFException e) {
                break;
            }
        }
    }

    private static void addHeaders(HttpUriRequest req, Map<String, String> headers, boolean dump) {
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            if (dump) {
                System.out.println("> " + entry.getKey() + ": " + entry.getValue());
            }
            req.addHeader(entry.getKey(), entry.getValue());
        }
    }

    public static Map<String, List<String>> parseQueryString(String queryString) {
        Map<String, List<String>> parameters = new LinkedHashMap<>(); // NOPMD
        if (StringUtils.isEmpty(queryString)) {
            return parameters;
        }

        for (String pair : queryString.split("&")) {
            String[] parameter = pair.split("=", 2);
            List<String> list = parameters.computeIfAbsent(parameter[0], k -> new ArrayList<>());
            if (parameter.length > 1) {
                list.add(parameter[1]);
            } else {
                list.add(null);
            }
        }
        return parameters;
    }

    private static CloseableHttpClient getHttpClient(boolean insecure, int timeout) {
        RequestConfig config =
                RequestConfig.custom()
                        .setConnectTimeout(timeout)
                        .setConnectionRequestTimeout(timeout)
                        .setSocketTimeout(timeout)
                        .build();
        if (insecure) {
            try {
                SSLContext context =
                        SSLContextBuilder.create()
                                .loadTrustMaterial(TrustAllStrategy.INSTANCE)
                                .build();

                HostnameVerifier verifier = new NoopHostnameVerifier();
                SSLConnectionSocketFactory factory =
                        new SSLConnectionSocketFactory(context, verifier);

                return HttpClients.custom()
                        .setDefaultRequestConfig(config)
                        .setSSLSocketFactory(factory)
                        .build();
            } catch (GeneralSecurityException e) {
                throw new AssertionError(e);
            }
        }
        return HttpClients.custom().setDefaultRequestConfig(config).build();
    }

    private static HttpUriRequest createHttpRequest(String method, URI uri, byte[] data) {
        HttpUriRequest request;

        if (HttpPost.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpPost(uri);
        } else if (HttpPut.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpPut(uri);
        } else if (HttpDelete.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpDelete(uri);
        } else if (HttpGet.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpGet(uri);
        } else if (HttpHead.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpHead(uri);
        } else {
            throw new IllegalArgumentException("Invalid method: " + method);
        }

        if (request instanceof HttpEntityEnclosingRequestBase && data != null) {
            ByteArrayEntity entity = new ByteArrayEntity(data);
            ((HttpEntityEnclosingRequestBase) request).setEntity(entity);
        }

        return request;
    }

    static void updateTokenCount(
            List<? extends CharSequence> list, AtomicInteger tokens, SignableRequest request) {
        tokens.addAndGet(TokenUtils.countTokens(list));
        if (System.getenv("EXCLUDE_INPUT_TOKEN") != null) {
            tokens.addAndGet(-request.getInputTokens());
        }
    }
}
