package com.amazonaws.jarjar;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;

public class JarConnection extends URLConnection {

    private URL url;

    public JarConnection(URL url) {
        super(url);
        this.url = url;
    }

    @Override
    public void connect() throws IOException {}

    @Override
    public InputStream getInputStream() throws IOException {
        String path = url.getFile();
        return JarConnection.class.getResourceAsStream(path);
    }
}
