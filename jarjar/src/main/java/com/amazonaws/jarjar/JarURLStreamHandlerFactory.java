package com.amazonaws.jarjar;

import java.net.URLStreamHandler;
import java.net.URLStreamHandlerFactory;

public class JarURLStreamHandlerFactory implements URLStreamHandlerFactory {

    private URLStreamHandlerFactory factory;

    public JarURLStreamHandlerFactory(URLStreamHandlerFactory factory) {
        this.factory = factory;
    }

    @Override
    public URLStreamHandler createURLStreamHandler(String protocol) {
        if ("jarjar".equals(protocol)) {
            return new JarURLStreamHandler();
        }
        if (factory != null) {
            return factory.createURLStreamHandler(protocol);
        }
        return null;
    }
}
