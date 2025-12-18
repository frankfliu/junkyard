package com.amazonaws.awscurl;

import ai.djl.util.Utils;

import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import javax.net.ssl.SSLSession;

@SuppressWarnings("PMD.SystemPrintln")
final class Client {

    private static final String DEFAULT_CONTENT_TYPE =
            Utils.getEnvOrSystemProperty("DEFAULT_CONTENT_TYPE");

    private HttpClient client;

    private Client(HttpClient client) {
        this.client = client;
    }

    public Response sendRequest(
            SignableRequest request,
            OutputStream ps,
            boolean dumpHeader,
            AtomicInteger tokens,
            long[] requestTime,
            String[] jq)
            throws IOException, InterruptedException {
        ps.write(
                ("\ntimestamp: " + System.currentTimeMillis() + "\ninput: ")
                        .getBytes(StandardCharsets.UTF_8));
        ps.write(request.getContent());
        ps.write(("\noutput: ").getBytes(StandardCharsets.UTF_8));
        long begin = System.nanoTime();
        if (dumpHeader) {
            String path = request.getUri().getPath();
            if (!path.startsWith("/") && !path.isEmpty()) {
                path = path.substring(1);
            }
            System.out.println("> " + request.getHttpMethod() + " /" + path + " HTTP/1.1");
            System.out.println("> ");
        }
        Map<String, String> map = new ConcurrentHashMap<>(request.getHeaders());
        map.putAll(request.getSignedHeaders());
        map.remove("Host");

        HttpRequest req =
                createHttpRequest(
                        request.getHttpMethod(),
                        request.getUri(),
                        request.getContent(),
                        map,
                        dumpHeader);

        HttpResponse<InputStream> resp =
                client.send(req, HttpResponse.BodyHandlers.ofInputStream());
        int code = resp.statusCode();
        HttpHeaders headers = resp.headers();
        if (dumpHeader) {
            System.out.println("> ");
            System.out.println("< HTTP/1.1 " + code);
            System.out.println("< ");
            for (Map.Entry<String, List<String>> header : headers.map().entrySet()) {
                System.out.println("< " + header.getKey() + ": " + header.getValue());
            }
            System.out.println("< ");
        }

        if (code >= 300 && ps instanceof NullOutputStream) {
            String body = Utils.toString(resp.body());
            System.out.println("HTTP error (" + resp.statusCode() + "): " + body);
            return new Response(code, body, headers);
        }

        List<String> ctHeader = headers.allValues("Content-Type");
        String contentType = DEFAULT_CONTENT_TYPE;
        for (String header : ctHeader) {
            String[] parts = header.split(";");
            contentType = parts[0];
            if ("text/event-stream".equals(contentType)) {
                break;
            }
        }

        Response ret;
        try (FirstByteCounterInputStream is = new FirstByteCounterInputStream(resp.body())) {
            if (tokens != null) {
                JsonUtils.resetException();
                if (contentType == null || "text/plain".equals(contentType)) {
                    String body = Utils.toString(is);
                    ret = new Response(code, body, headers);
                    ps.write(body.getBytes(StandardCharsets.UTF_8));
                    updateTokenCount(Collections.singletonList(body), tokens, request);
                } else if ("application/json".equals(contentType)) {
                    String body = Utils.toString(is);
                    ret = new Response(code, body, headers);
                    ps.write(body.getBytes(StandardCharsets.UTF_8));
                    try {
                        JsonElement element = JsonUtils.GSON.fromJson(body, JsonElement.class);
                        List<String> lines = new ArrayList<>();
                        JsonUtils.getJsonList(element, lines, jq);
                        updateTokenCount(lines, tokens, request);
                    } catch (JsonParseException e) {
                        AwsCurl.logger.warn("Invalid json response: {}", body);
                        ret = new Response(500, null, headers);
                    }
                } else if ("application/jsonlines".equals(contentType)) {
                    boolean hasError = false;
                    try (BufferedReader reader =
                            new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                        String line;
                        List<StringBuilder> list = new ArrayList<>();
                        while ((line = reader.readLine()) != null) {
                            hasError = JsonUtils.processJsonLine(list, ps, line, jq) || hasError;
                        }
                        updateTokenCount(list, tokens, request);
                        ret = new Response(code, String.join("", list), headers);
                    }
                    if (hasError) {
                        ret = new Response(500, null, headers);
                    }
                } else if ("text/event-stream".equals(contentType)) {
                    String body =
                            handleServerSentEvent(is, requestTime, begin, jq, tokens, request, ps);
                    ret = new Response(code, body, headers);
                } else if ("application/vnd.amazon.eventstream".equals(contentType)) {
                    String realContentType =
                            headers.firstValue("X-Amzn-SageMaker-Content-Type").orElse(null);
                    String body =
                            handleEventStream(
                                    is,
                                    ps,
                                    realContentType,
                                    requestTime,
                                    begin,
                                    jq,
                                    tokens,
                                    request);
                    ret = new Response(code, body, headers);
                } else {
                    String body = Utils.toString(is);
                    ret = new Response(code, body, headers);
                }
            } else if ("application/vnd.amazon.eventstream".equals(contentType)) {
                String realContentType =
                        headers.firstValue("X-Amzn-SageMaker-Content-Type").orElse(null);
                String body =
                        handleEventStream(
                                is, ps, realContentType, requestTime, begin, jq, null, request);
                ret = new Response(code, body, headers);
            } else {
                byte[] body = Utils.toByteArray(is);
                ps.write(body);
                ps.flush();
                ret = new Response(code, new String(body, StandardCharsets.UTF_8), headers);
            }
            requestTime[0] += System.nanoTime() - begin;
            if (requestTime[1] == -1) {
                requestTime[1] = is.getTimeToFirstByte() - begin;
            }

            return ret;
        }
    }

    private static String handleServerSentEvent(
            InputStream is,
            long[] requestTime,
            long begin,
            String[] jq,
            AtomicInteger tokens,
            SignableRequest request,
            OutputStream ps)
            throws IOException {
        List<String> list = new ArrayList<>();
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.startsWith("data:")) {
                    continue;
                }
                if (requestTime[1] == -1) {
                    requestTime[1] = System.nanoTime() - begin;
                }
                line = line.substring(5);
                ps.write(line.getBytes(StandardCharsets.UTF_8));
                ps.write(new byte[] {'\n'});
                JsonElement element = JsonUtils.GSON.fromJson(line, JsonElement.class);
                JsonUtils.getJsonList(element, list, jq);
            }
            if (tokens != null) {
                updateTokenCount(list, tokens, request);
            }
        }
        return String.join("", list);
    }

    private static String handleEventStream(
            InputStream is,
            OutputStream ps,
            String realContentType,
            long[] requestTime,
            long begin,
            String[] jq,
            AtomicInteger tokens,
            SignableRequest request)
            throws IOException {
        byte[] buf = new byte[12];
        byte[] payload = new byte[512];
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        while (true) {
            if (is.readNBytes(buf, 0, buf.length) == 0) {
                break;
            }
            ByteBuffer bb = ByteBuffer.wrap(buf);
            bb.order(ByteOrder.BIG_ENDIAN);
            int totalLength = bb.getInt();
            int headerLength = bb.getInt();
            int payloadLength = totalLength - headerLength - 12 - 4;
            int size = totalLength - 12;
            if (size > payload.length) {
                payload = new byte[size];
            }
            if (is.readNBytes(payload, 0, size) == 0) {
                break;
            }
            if (payloadLength == 0) {
                break;
            }
            bos.write(payload, headerLength, payloadLength);
        }
        bos.close();

        byte[] bytes = bos.toByteArray();
        InputStream bis = new ByteArrayInputStream(bytes);
        if (realContentType != null) {
            realContentType = realContentType.split(";")[0];
        } else {
            realContentType = DEFAULT_CONTENT_TYPE;
        }
        if ("text/event-stream".equalsIgnoreCase(realContentType)) {
            String body = handleServerSentEvent(bis, requestTime, begin, jq, tokens, request, ps);
            requestTime[1] = -1; // rely on FirstByteCounterInputStream
            return body;
        }
        List<StringBuilder> list = new ArrayList<>();
        Scanner scanner = new Scanner(bis, StandardCharsets.UTF_8);
        while (scanner.hasNext()) {
            String line = scanner.nextLine();
            if (JsonUtils.processJsonLine(list, ps, line, jq)) {
                throw new IOException("Response contains error");
            }
        }
        scanner.close();

        if (tokens != null) {
            updateTokenCount(list, tokens, request);
        }
        return String.join("", list);
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

    public static Client getHttpClient(int timeout) {
        return new Client(
                HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(timeout)).build());
    }

    private static HttpRequest createHttpRequest(
            String method, URI uri, byte[] data, Map<String, String> headers, boolean dump) {
        HttpRequest.Builder builder = HttpRequest.newBuilder().uri(uri);
        switch (method) {
            case "GET":
                builder.GET();
                break;
            case "POST":
                builder.POST(HttpRequest.BodyPublishers.ofByteArray(data));
                break;
            case "PUT":
                builder.PUT(HttpRequest.BodyPublishers.ofByteArray(data));
                break;
            case "DELETE":
                builder.DELETE();
                break;
            case "HEAD":
                builder.method("HEAD", HttpRequest.BodyPublishers.noBody());
                break;
            default:
                throw new IllegalArgumentException("Invalid method: " + method);
        }
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            if (dump) {
                System.out.println("> " + entry.getKey() + ": " + entry.getValue());
            }
            builder.header(entry.getKey(), entry.getValue());
        }
        return builder.build();
    }

    static void updateTokenCount(
            List<? extends CharSequence> list, AtomicInteger tokens, SignableRequest request) {
        tokens.addAndGet(TokenUtils.countTokens(list));
        if (Utils.getEnvOrSystemProperty("EXCLUDE_INPUT_TOKEN") != null) {
            tokens.addAndGet(-request.getInputTokens());
        }
    }

    private static final class FirstByteCounterInputStream extends InputStream {

        private long timeToFirstByte;
        private InputStream is;

        FirstByteCounterInputStream(InputStream is) {
            this.is = is;
        }

        long getTimeToFirstByte() {
            return timeToFirstByte;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] b) throws IOException {
            int read = is.read(b);
            if (timeToFirstByte == 0 && read > 0) {
                timeToFirstByte = System.nanoTime();
            }
            return read;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int read = is.read(b, off, len);
            if (timeToFirstByte == 0 && read > 0) {
                timeToFirstByte = System.nanoTime();
            }
            return read;
        }

        /** {@inheritDoc} */
        @Override
        public int read() throws IOException {
            int read = is.read();
            if (timeToFirstByte == 0 && read > 0) {
                timeToFirstByte = System.nanoTime();
            }
            return read;
        }

        /** {@inheritDoc} */
        @Override
        public void close() throws IOException {
            is.close();
        }
    }

    static final class ErrorHttpResponse implements HttpResponse<InputStream> {

        /** {@inheritDoc} */
        @Override
        public int statusCode() {
            return 500;
        }

        /** {@inheritDoc} */
        @Override
        public HttpRequest request() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public Optional<HttpResponse<InputStream>> previousResponse() {
            return Optional.empty();
        }

        /** {@inheritDoc} */
        @Override
        public HttpHeaders headers() {
            return HttpHeaders.of(Collections.emptyMap(), (k, v) -> true);
        }

        /** {@inheritDoc} */
        @Override
        public InputStream body() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public Optional<SSLSession> sslSession() {
            return Optional.empty();
        }

        /** {@inheritDoc} */
        @Override
        public URI uri() {
            return null;
        }

        /** {@inheritDoc} */
        @Override
        public HttpClient.Version version() {
            return null;
        }
    }
}
