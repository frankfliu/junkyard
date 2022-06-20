package com.amazonaws.jarjar;

import java.net.URL;
import java.net.URLConnection;
import java.net.URLStreamHandler;

public class JarURLStreamHandler extends URLStreamHandler {

    @Override
    protected URLConnection openConnection(URL u) {
        return new JarConnection(u);
    }
}
