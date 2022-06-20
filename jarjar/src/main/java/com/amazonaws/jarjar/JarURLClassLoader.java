package com.amazonaws.jarjar;

import java.lang.reflect.Field;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.net.URLStreamHandlerFactory;
import java.util.List;

public class JarURLClassLoader extends URLClassLoader {

    static {
        try {
            URL.setURLStreamHandlerFactory(new JarURLStreamHandlerFactory(null));
        } catch (Error error) {
            try {
                Field f = URL.class.getDeclaredField("factory");
                f.setAccessible(true); // NOPMD
                URLStreamHandlerFactory factory = (URLStreamHandlerFactory) f.get(null);
                f.set(null, new JarURLStreamHandlerFactory(factory));
            } catch (ReflectiveOperationException e) {
                throw error;
            }
        }
    }

    public JarURLClassLoader(List<String> jars) {
        super(new URL[] {}, Thread.currentThread().getContextClassLoader());
        try {
            for (String fileName : jars) {
                addURL(new URL("jarjar:/" + fileName));
            }
        } catch (MalformedURLException e) {
            e.printStackTrace(); // NOPMD
        }
    }
}
