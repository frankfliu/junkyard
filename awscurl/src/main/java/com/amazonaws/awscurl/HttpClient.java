package com.amazonaws.awscurl;

import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.training.util.DownloadUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.NullOutputStream;
import org.apache.http.Header;
import org.apache.http.HttpResponse;
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
import org.apache.http.ssl.SSLContextBuilder;
import org.apache.http.util.EntityUtils;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.net.URI;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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

    static final Gson GSON = new Gson();
    private static final Type LIST_TYPE = new TypeToken<List<Map<String, String>>>() {}.getType();
    private static final Type MAP_TYPE = new TypeToken<Map<String, List<String>>>() {}.getType();
    private static final HuggingFaceTokenizer tokenizer = getTokenizer();

    private HttpClient() {}

    public static HttpResponse sendRequest(
            SignableRequest request,
            boolean insecure,
            int timeout,
            OutputStream ps,
            boolean dumpHeader,
            AtomicInteger tokens,
            long[] firstToken)
            throws IOException {
        try (CloseableHttpClient client = getHttpClient(insecure, timeout)) {
            HttpUriRequest req =
                    createHttpRequest(
                            request.getHttpMethod(), request.getUri(), request.getContent());
            if (dumpHeader) {
                String path = request.getUri().getPath();
                if (!path.startsWith("/")) {
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
                    countTokens(Collections.singletonList(body), tokens, request);
                    return resp;
                } else if ("application/json".equals(contentType)) {
                    String body = EntityUtils.toString(resp.getEntity());
                    ps.write(body.getBytes(StandardCharsets.UTF_8));

                    List<Map<String, String>> list = GSON.fromJson(body, LIST_TYPE);
                    List<String> lines = new ArrayList<>();
                    for (Map<String, String> item : list) {
                        String text = item.get("generated_text");
                        if (text != null) {
                            lines.add(text);
                        }
                    }
                    countTokens(lines, tokens, request);
                    return resp;
                } else if ("application/jsonlines".equals(contentType)) {
                    InputStream is = resp.getEntity().getContent();
                    try (BufferedReader reader =
                            new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                        String line;
                        List<StringBuilder> list = new ArrayList<>();
                        while ((line = reader.readLine()) != null) {
                            processJsonLine(list, firstToken, ps, line);
                        }
                        countTokens(list, tokens, request);
                    }
                    return resp;
                } else if ("application/vnd.amazon.eventstream".equals(contentType)) {
                    List<StringBuilder> list = new ArrayList<>();
                    InputStream is = resp.getEntity().getContent();
                    handleEventStream(is, list, firstToken, ps);
                    countTokens(list, tokens, request);
                    return resp;
                }
            }

            try (InputStream is = resp.getEntity().getContent()) {
                if ("application/vnd.amazon.eventstream".equals(contentType)) {
                    List<StringBuilder> list = new ArrayList<>();
                    handleEventStream(is, list, firstToken, ps);
                } else {
                    IOUtils.copy(is, ps);
                    ps.flush();
                }
            }
            return resp;
        }
    }

    private static HuggingFaceTokenizer getTokenizer() {
        try {
            Path cacheDir = Utils.getEngineCacheDir("tokenizers");
            Platform platform = Platform.detectPlatform("tokenizers");
            String classifier = platform.getClassifier();
            String version = platform.getVersion();
            Path dir = cacheDir.resolve(version + '-' + classifier);
            String libName = System.mapLibraryName("tokenizers");
            Path path = dir.resolve(libName);
            if (!Files.exists(path)) {
                Files.createDirectories(dir);
                String djlVersion = Engine.getDjlVersion();
                String url =
                        "https://publish.djl.ai/tokenizers/"
                                + version.split("-")[0]
                                + "/jnilib/"
                                + djlVersion
                                + '/'
                                + classifier
                                + '/'
                                + libName;
                DownloadUtils.download(new URL(url), path, null);
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to load HuggingFace tokenizer.", e);
        }

        HuggingFaceTokenizer.Builder builder = HuggingFaceTokenizer.builder();
        String name = System.getenv("TOKENIZER");
        if (name != null) {
            Path path = Paths.get(name);
            if (Files.exists(path)) {
                builder.optTokenizerPath(path);
            } else {
                builder.optTokenizerName(name);
            }
            try {
                return builder.build();
            } catch (Exception e) {
                System.out.println(
                        "Invalid tokenizer: "
                                + name
                                + ", please unset environment variable TOKENIZER if don't want to"
                                + " use tokenizer");
            }
        }
        return null;
    }

    private static void handleEventStream(
            InputStream is, List<StringBuilder> list, long[] firstToken, OutputStream ps)
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
                processJsonLine(list, firstToken, ps, line);
            } catch (EOFException e) {
                break;
            }
        }
    }

    private static void processJsonLine(
            List<StringBuilder> list, long[] firstToken, OutputStream ps, String line)
            throws IOException {
        boolean first = firstToken[0] == 0L;
        if (first) {
            firstToken[0] = System.nanoTime();
        }
        try {
            Map<String, List<String>> map = GSON.fromJson(line, MAP_TYPE);
            List<String> item = map.get("outputs");
            if (item != null) {
                if (list.isEmpty()) {
                    for (String s : item) {
                        list.add(new StringBuilder(s));
                    }
                } else {
                    for (int i = 0; i < item.size(); ++i) {
                        list.get(i).append(item.get(i));
                    }
                }
            }
        } catch (JsonParseException e) {
            if (first) {
                System.out.println("Invalid json line: " + line);
            }
            list.add(new StringBuilder(line));
        }

        ps.write(line.getBytes(StandardCharsets.UTF_8));
        ps.write(new byte[] {'\n'});
    }

    private static void countTokens(
            List<? extends CharSequence> list, AtomicInteger tokens, SignableRequest request) {
        for (CharSequence item : list) {
            if (tokenizer != null) {
                Encoding encoding = tokenizer.encode(item.toString());
                tokens.addAndGet(encoding.getIds().length);
            } else {
                String[] token = item.toString().split("\\s");
                tokens.addAndGet(token.length);
            }
        }
        if (System.getenv("EXCLUDE_INPUT_TOKEN") != null) {
            tokens.addAndGet(-request.getInputTokens());
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
}
