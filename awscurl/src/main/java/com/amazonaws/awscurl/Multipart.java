package com.amazonaws.awscurl;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Multipart {

    private static final String HYPHEN = "--";
    private static final String CRLF = "\r\n";
    private static final String CHARS =
            "-_1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    private static final Random RAND = new Random();

    private List<BodyPart> parts;
    private String contentType;
    private String multipartBoundary;

    public Multipart() {
        multipartBoundary = generateMultipartBoundary();
        contentType = "multipart/form-data; boundary=" + multipartBoundary + "; charset=UTF-8";
        parts = new ArrayList<>();
    }

    public String getContentType() {
        return contentType;
    }

    public void addBodyPart(String name, byte[] content, String contentType) {
        parts.add(new BodyPart(name, content, contentType));
    }

    public void addBodyPart(String name, Path file, String contentType, String fileName)
            throws IOException {
        byte[] content = Files.readAllBytes(file);
        parts.add(new BodyPart(name, content, contentType, fileName));
    }

    public void writeTo(OutputStream os) throws IOException {
        PrintStream ps = new PrintStream(os, false, StandardCharsets.UTF_8);
        ps.append(HYPHEN).append(multipartBoundary);

        if (parts.isEmpty()) {
            ps.append(CRLF);
            // put a empty line to finish bodyPart header
            ps.append(CRLF);
            ps.append(CRLF).append(HYPHEN).append(multipartBoundary);
        } else {
            for (BodyPart part : parts) {
                ps.append(CRLF);
                part.writeTo(ps);
                ps.append(CRLF).append(HYPHEN).append(multipartBoundary);
            }
        }
        ps.append(HYPHEN).append(CRLF);
        ps.flush();
    }

    private static String generateMultipartBoundary() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("awscurl-");
        for (int i = 0; i < 20; i++) {
            sb.append(CHARS.charAt(RAND.nextInt(64)));
        }
        return sb.toString();
    }

    private static class BodyPart {

        private byte[] content;
        private List<String> headers = new ArrayList<>();

        public BodyPart(String name, byte[] content, String contentType) {
            this(name, content, contentType, null);
        }

        public BodyPart(String name, byte[] content, String contentType, String fileName) {
            this.content = content;
            headers.add("Content-Type: " + contentType);
            if (fileName == null) {
                headers.add("Content-Disposition: form-data; name=\"" + name + "\"");
            } else {
                headers.add(
                        "Content-Disposition: form-data; name=\""
                                + name
                                + "\"; filename=\""
                                + fileName
                                + "\"");
                headers.add("Content-Transfer-Encoding: binary");
            }
        }

        public void writeTo(PrintStream ps) throws IOException {
            for (String header : headers) {
                ps.append(header).append(CRLF);
            }
            ps.print(CRLF); // End of header
            ps.write(content);
        }
    }
}
