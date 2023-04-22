package com.amazonaws.awscurl;

import java.net.URI;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class SignableRequest {

    private String serviceName;
    private String httpMethod;
    private URI uri;
    private String path;
    private Map<String, String> headers;
    private Map<String, String> signedHeaders;
    private Map<String, List<String>> parameters;
    private AWS4Signer signer;

    private byte[] content;
    private int timeOffset;
    private long signingTime;

    public SignableRequest(String serviceName, URI uri) {
        this.serviceName = serviceName;
        httpMethod = "POST";
        headers = new ConcurrentHashMap<>();
        parameters = new LinkedHashMap<>();
        headers.put("User-Agent", "awscurl/1.0.0");
        headers.put("Accept", "*/*");
        signedHeaders = new ConcurrentHashMap<>();
        setUri(uri);
    }

    private SignableRequest() {}

    public void sign() {
        if (signer == null) {
            return;
        }
        long now = System.nanoTime();
        if (now - signingTime > 2 * 60000000000L) {
            signingTime = now;
            signer.sign(this);
        }
    }

    public SignableRequest copy() {
        SignableRequest req = new SignableRequest();
        req.serviceName = serviceName;
        req.uri = uri;
        req.httpMethod = httpMethod;
        req.path = path;
        req.headers = new ConcurrentHashMap<>(headers);
        req.headers.remove("X-Amzn-SageMaker-Custom-Attributes");
        req.signedHeaders = new ConcurrentHashMap<>();
        req.parameters = new ConcurrentHashMap<>(parameters);
        req.signer = signer;
        req.content = content;
        req.timeOffset = timeOffset;
        return req;
    }

    public String getServiceName() {
        return serviceName;
    }

    public void setSigner(AWS4Signer signer) {
        this.signer = signer;
    }

    private void setUri(URI uri) {
        this.uri = uri;
        String schema = uri.getScheme().toLowerCase(Locale.ENGLISH);
        int port = uri.getPort();
        String host = null;
        if (port != -1) {
            int defaultPort;
            if ("https".equals(schema)) {
                defaultPort = 443;
            } else if ("http".equals(schema)) {
                defaultPort = 80;
            } else {
                defaultPort = -1;
            }
            if (port != defaultPort) {
                host = uri.getHost() + ':' + port;
            }
        }
        if (host == null) {
            host = uri.getHost();
        }
        path = uri.getPath();
        parameters = HttpClient.parseQueryString(uri.getQuery());
        headers.put("Host", host);
    }

    public URI getUri() {
        return uri;
    }

    public String getHttpMethod() {
        return httpMethod;
    }

    public void setHttpMethod(String httpMethod) {
        this.httpMethod = httpMethod;
    }

    public String getPath() {
        return path;
    }

    public void setHeaders(Map<String, String> headers) {
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            addHeader(entry.getKey(), entry.getValue());
        }
    }

    void addHeader(String name, String value) {
        for (String key : headers.keySet()) {
            if (key.equalsIgnoreCase(name)) {
                headers.remove(key);
                break;
            }
        }
        headers.put(name, value);
    }

    public Map<String, String> getHeaders() {
        return headers;
    }

    public Map<String, String> getSignedHeaders() {
        return signedHeaders;
    }

    public void setSignedHeaders(Map<String, String> signedHeaders) {
        this.signedHeaders = signedHeaders;
    }

    public void setParameters(Map<String, List<String>> parameters) {
        this.parameters.clear();
        this.parameters.putAll(parameters);
    }

    public void addParameter(String name, String value) {
        List<String> paramList = parameters.computeIfAbsent(name, k -> new ArrayList<>());
        paramList.add(value);
    }

    public void addParameters(String name, List<String> values) {
        for (String value : values) {
            addParameter(name, value);
        }
    }

    public Map<String, List<String>> getParameters() {
        return parameters;
    }

    public boolean notHasContent() {
        return content == null || content.length == 0;
    }

    public byte[] getContent() {
        if (content == null) {
            content = new byte[0];
        }
        return content;
    }

    public void setContent(byte[] content) {
        this.content = content;
    }

    public int getTimeOffset() {
        return timeOffset;
    }

    public void setTimeOffset(int timeOffset) {
        this.timeOffset = timeOffset;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(uri.toString()).append(' ');
        if (!getHeaders().isEmpty()) {
            builder.append("Headers: (");
            for (String key : getHeaders().keySet()) {
                String value = getHeaders().get(key);
                builder.append(key).append(": ").append(value).append(", ");
            }
            builder.append(')');
        }

        return builder.toString();
    }
}
