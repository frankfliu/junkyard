package com.amazonaws.jarjar;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.jar.JarInputStream;
import java.util.jar.Manifest;

public final class Bootstrap {

    private Bootstrap() {}

    @SuppressWarnings({"PMD.SystemPrintln", "PMD.AvoidPrintStackTrace"})
    public static void main(String[] args) throws IOException {
        ProtectionDomain cd = Bootstrap.class.getProtectionDomain();
        URL url = cd.getCodeSource().getLocation();
        String path = url.getPath();
        if (!"file".equalsIgnoreCase(url.getProtocol()) || !path.toLowerCase().endsWith(".jar")) {
            System.out.println("Must run in a jar file.");
            return;
        }

        String mainClass = System.getProperty("main");
        List<String> jars = new ArrayList<>();

        try (JarFile jarFile = new JarFile(path)) {
            Enumeration<JarEntry> en = jarFile.entries();
            while (en.hasMoreElements()) {
                JarEntry entry = en.nextElement();
                String fileName = entry.getName();
                if (fileName.endsWith(".jar")) {
                    jars.add(fileName);
                    if (mainClass == null) {
                        JarInputStream jis = new JarInputStream(jarFile.getInputStream(entry));
                        Manifest manifest = jis.getManifest();
                        Attributes attr = manifest.getMainAttributes();
                        mainClass = attr.getValue("Main-Class");
                        jis.close();
                    }
                }
            }
        }

        if (mainClass == null) {
            System.out.println("No Main-Class defined.");
            return;
        }

        JarURLClassLoader cl = new JarURLClassLoader(jars);
        try {
            Class<?> clazz = Class.forName(mainClass, false, cl);
            Method mainMethod = clazz.getMethod("main", String[].class);
            mainMethod.invoke(null, (Object) args);
        } catch (ClassNotFoundException e) {
            System.out.println("Class not found: " + mainClass);
            if (Boolean.getBoolean("DEBUG")) {
                e.printStackTrace();
            }
        } catch (NoSuchMethodException e) {
            System.out.println("No main method found in: " + mainClass);
            if (Boolean.getBoolean("DEBUG")) {
                e.printStackTrace();
            }
        } catch (InvocationTargetException e) {
            System.out.println("Failed in invoke main in: " + mainClass);
            if (Boolean.getBoolean("DEBUG")) {
                e.printStackTrace();
            }
        } catch (IllegalAccessException e) {
            System.out.println("main method can not be accessed in: " + mainClass);
            if (Boolean.getBoolean("DEBUG")) {
                e.printStackTrace();
            }
        }
    }
}
