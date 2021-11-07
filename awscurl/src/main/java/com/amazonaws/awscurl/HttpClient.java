package com.amazonaws.awscurl;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import org.apache.commons.io.IOUtils;
import org.apache.http.Header;
import org.apache.http.HttpResponse;
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

@SuppressWarnings("PMD.SystemPrintln")
public final class HttpClient {

    private HttpClient() {}

    public static int sendRequest(
            SignableRequest request,
            boolean insecure,
            int timeOut,
            OutputStream ps,
            boolean dumpHeader)
            throws IOException {
        try (CloseableHttpClient client = getHttpClient(insecure)) {
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

            Map<String, String> headers = request.getHeaders();
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                if (dumpHeader) {
                    System.out.println("> " + entry.getKey() + ": " + entry.getValue());
                }
                req.addHeader(entry.getKey(), entry.getValue());
            }

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

            try (InputStream is = resp.getEntity().getContent()) {
                IOUtils.copy(is, ps);
                ps.flush();
            }
            return code;
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

    private static CloseableHttpClient getHttpClient(boolean insecure) {
        if (insecure) {
            try {
                SSLContext context =
                        SSLContextBuilder.create()
                                .loadTrustMaterial(TrustAllStrategy.INSTANCE)
                                .build();

                HostnameVerifier verifier = new NoopHostnameVerifier();
                SSLConnectionSocketFactory factory =
                        new SSLConnectionSocketFactory(context, verifier);

                return HttpClients.custom().setSSLSocketFactory(factory).build();
            } catch (GeneralSecurityException e) {
                throw new AssertionError(e);
            }
        }
        return HttpClients.createDefault();
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
