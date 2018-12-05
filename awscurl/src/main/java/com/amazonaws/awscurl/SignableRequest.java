package com.amazonaws.awscurl;

import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class SignableRequest {

    private String serviceName;
    private String httpMethod;
    private URI uri;
    private String host;
    private String path;
    private Map<String, String> headers;
    private Map<String, List<String>> parameters;

    private byte[] content;
    private int timeOffset;

    public SignableRequest(String serviceName) {
        this.serviceName = serviceName;
        httpMethod = "POST";
        headers = new HashMap<>();
        parameters = new LinkedHashMap<>();
        headers.put("User-Agent", "awscurl/1.0.0");
        headers.put("Accept", "*/*");
    }

    public String getServiceName() {
        return serviceName;
    }

    public void setUri(URI uri) {
        this.uri = uri;
        String schema = uri.getScheme().toLowerCase(Locale.ENGLISH);
        int port = uri.getPort();
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

    public String getHost() {
        return host;
    }

    public String getPath() {
        return path;
    }

    public void setHeaders(Map<String, String> headers) {
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            addHeader(entry.getKey(), entry.getValue());
        }
    }

    public void addHeader(String name, String value) {
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
