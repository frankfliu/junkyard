package com.amazonaws.awscurl;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.NullOutputStream;
import org.apache.http.HttpEntity;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;

@SuppressWarnings("PMD.SystemPrintln")
public final class AwsCurl {

    private AwsCurl() {}

    public static void main(String[] args) {
        String jarName = getJarName();

        Options options = Config.getOptions();
        DefaultParser parser = new DefaultParser();
        try {
            if (args.length == 0
                    || args[0].equalsIgnoreCase("-h")
                    || args[0].equalsIgnoreCase("--help")) {
                printHelp(jarName, options);
                return;
            }

            CommandLine cmd = parser.parse(options, args, null, false);
            List<String> cmdArgs = cmd.getArgList();
            if (cmdArgs.isEmpty()) {
                printHelp(jarName, options);
                return;
            }
            Config config = new Config(cmd);
            String url = config.getUrl(cmdArgs.get(0));
            URI uri;
            try {
                uri = new URI(url);
            } catch (URISyntaxException e) {
                System.err.println("Invalid url: " + url);
                return;
            }

            String serviceName = config.getServiceName();
            SignableRequest request = new SignableRequest(serviceName);
            request.setUri(uri);
            request.setContent(config.getRequestBody());
            request.setHeaders(config.getRequestHeaders());
            request.setHttpMethod(config.getRequestMethod());

            if (serviceName != null) {
                AWSCredentials credentials = AWSCredentials.getCredentials(config.getProfile());
                if (credentials == null) {
                    System.err.println(
                            "The config profile (" + config.getProfile() + ") could not be found.");
                    return;
                }
                if (StringUtils.isEmpty(credentials.getAWSAccessKeyId())) {
                    System.err.println("Anonymous credentials is not supported.");
                    return;
                }
                String region = config.getRegion();
                if (region == null) {
                    region = inferRegion(url, serviceName);
                    if (region == null) {
                        region = credentials.getRegion();
                        if (region == null) {
                            System.err.println("Not able to obtain region name from profile.");
                            return;
                        }
                    }
                }

                AWS4Signer signer = new AWS4Signer(serviceName, region);
                signer.sign(request, credentials);
            }

            boolean insecure = config.isInsecure();
            OutputStream os = config.getOutput();
            boolean printHeader = config.isInclude() || config.isVerbose();

            int clients = config.getClients();
            int nRequests = config.getNumberOfRequests();
            final Queue<Long> queue = new ConcurrentLinkedQueue<>();
            final Queue<Long> errors = new ConcurrentLinkedQueue<>();

            ExecutorService executor = Executors.newFixedThreadPool(clients);
            ArrayList<Callable<Void>> tasks = new ArrayList<>(clients);
            for (int i = 0; i < clients; ++i) {
                tasks.add(
                        () -> {
                            for (int j = 0; j < nRequests; ++j) {
                                long begin = System.nanoTime();
                                int code =
                                        HttpClient.sendRequest(
                                                request,
                                                insecure,
                                                config.getConnectTimeout(),
                                                os,
                                                printHeader);
                                if (code == 200) {
                                    queue.add(System.nanoTime() - begin);
                                } else {
                                    errors.add(System.nanoTime() - begin);
                                }
                            }
                            return null;
                        });
            }

            long start = System.nanoTime();
            executor.invokeAll(tasks);
            long delta = System.nanoTime() - start;

            executor.shutdown();

            if (nRequests > 1 && clients > 0) {
                int totalRequest = clients * nRequests;
                int successReq = queue.size();
                int errorReq = errors.size();
                List<Long> list = new ArrayList<>(queue);
                Collections.sort(list);
                long totalTime = list.stream().mapToLong(val -> val).sum();

                System.out.println(String.format("Total time: %.2f ms.", delta / 1000000d));
                System.out.println(
                        String.format(
                                "Non 200 responses: %d, error rate: %.2f",
                                errorReq, 100d * errorReq / totalRequest));
                System.out.println("Concurrent clients: " + clients);
                System.out.println("Total requests: " + totalRequest);
                if (successReq > 0) {
                    System.out.println(
                            String.format("TPS: %.2f/s", successReq * 1000000000d / delta));
                    System.out.println(
                            String.format(
                                    "Average Latency: %.2f ms.",
                                    totalTime / 1000000d / successReq));
                    System.out.println(
                            String.format("P50: %.2f ms.", list.get(successReq / 2) / 1000000d));
                    System.out.println(
                            String.format(
                                    "P90: %.2f ms.", list.get(successReq * 9 / 10) / 1000000d));
                    System.out.println(
                            String.format(
                                    "P99: %.2f ms.", list.get(successReq * 99 / 100) / 1000000d));
                }
            }
        } catch (IOException | InterruptedException e) {
            System.err.println(e.getMessage());
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            printHelp(jarName, options);
        }
    }

    private static void printHelp(String jarFileName, Options options) {
        String message;
        if (jarFileName == null) {
            message = "awscurl <URL>";
        } else {
            message = "java -jar " + jarFileName + " <URL>";
        }
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(message, options);
    }

    private static String getJarName() {
        URL url = AwsCurl.class.getProtectionDomain().getCodeSource().getLocation();
        String path = url.getPath();
        if ("file".equalsIgnoreCase(url.getProtocol())) {
            File file = new File(path);
            if (path.toLowerCase().endsWith(".jar")) { // we only support jar file for now
                return file.getName();
            }
        }
        return null;
    }

    private static String inferRegion(String url, String serviceName) {
        Pattern pattern =
                Pattern.compile(
                        "http(s)?://(.+\\.)?" + serviceName + "\\.(.+)\\.amazonaws\\.com(/.*)?",
                        Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(url);
        if (matcher.matches()) {
            return matcher.group(3);
        }

        pattern =
                Pattern.compile(
                        "http(s)?://(.+\\.)?(.+)\\." + serviceName + "\\.amazonaws\\.com(/.*)?",
                        Pattern.CASE_INSENSITIVE);
        matcher = pattern.matcher(url);
        if (matcher.matches()) {
            return matcher.group(3);
        }
        return null;
    }

    private static final class Config {

        private String serviceName;
        private String region;
        private String profile;
        private String contentType;
        private int connectTimeout;
        private String[] data;
        private String[] dataRaw;
        private String[] dataAscii;
        private String[] dataBinary;
        private String[] dataUrlencode;
        private String[] form;
        private String[] formString;
        private String requestMethod;
        private boolean forceGet;
        private String[] headers;
        private boolean include;
        private boolean insecure;
        private String referer;
        private String output;
        private String userAgent;
        private String uploadFile;
        private boolean verbose;
        private int nRequests;
        private int clients;

        public Config(CommandLine cmd) {
            serviceName = cmd.getOptionValue("service");
            region = cmd.getOptionValue("region");
            profile = cmd.getOptionValue("profile");
            try {
                if (cmd.hasOption("connect-timeout")) {
                    connectTimeout = Integer.parseInt(cmd.getOptionValue("connect-timeout")) * 1000;
                } else {
                    connectTimeout = 2000;
                }
            } catch (NumberFormatException e) {
                connectTimeout = 2000;
            }
            data = cmd.getOptionValues("data");
            dataRaw = cmd.getOptionValues("data-raw");
            dataAscii = cmd.getOptionValues("data-ascii");
            dataBinary = cmd.getOptionValues("data-binary");
            dataUrlencode = cmd.getOptionValues("data-urlencode");
            form = cmd.getOptionValues("form");
            formString = cmd.getOptionValues("form-string");
            if (cmd.hasOption("request")) {
                requestMethod = cmd.getOptionValue("request");
            }
            forceGet = cmd.hasOption("get");
            headers = cmd.getOptionValues("header");
            include = cmd.hasOption("include");
            insecure = cmd.hasOption("insecure");
            output = cmd.getOptionValue("output");
            referer = cmd.getOptionValue("referer");
            output = cmd.getOptionValue("output");
            uploadFile = cmd.getOptionValue("upload-file");
            userAgent = cmd.getOptionValue("user-agent");
            verbose = cmd.hasOption("verbose");
            if (cmd.hasOption("repeat")) {
                try {
                    nRequests = Integer.parseInt(cmd.getOptionValue("repeat"));
                } catch (NumberFormatException e) {
                    nRequests = 1;
                }
            } else {
                nRequests = 1;
            }
            if (cmd.hasOption("clients")) {
                try {
                    clients = Integer.parseInt(cmd.getOptionValue("clients"));
                } catch (NumberFormatException e) {
                    clients = 1;
                }
            } else {
                clients = 1;
            }
        }

        public static Options getOptions() {
            Options options = new Options();
            options.addOption(
                    Option.builder("n")
                            .longOpt("service")
                            .hasArg()
                            .argName("SERVICE")
                            .desc("AWS service name")
                            .build());
            options.addOption(
                    Option.builder("r")
                            .longOpt("region")
                            .hasArg()
                            .argName("REGION")
                            .desc("AWS region name")
                            .build());
            options.addOption(
                    Option.builder("p")
                            .longOpt("profile")
                            .hasArg()
                            .argName("PROFILE")
                            .desc("AWS credentials profile name")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("compressed")
                            .desc("Request compressed response (using deflate or gzip)")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("connect-timeout")
                            .hasArg()
                            .argName("SECONDS")
                            .desc("Maximum time allowed for connection")
                            .build());
            options.addOption(
                    Option.builder("d")
                            .longOpt("data")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST data")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("data-raw")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST data, '@' allowed")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("data-ascii")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST ASCII data")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("data-binary")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST binary data")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("data-urlencode")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST data url encoded")
                            .build());
            options.addOption(
                    Option.builder("F")
                            .longOpt("form")
                            .hasArgs()
                            .argName("CONTENT")
                            .desc("Specify HTTP multipart POST data")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("form-string")
                            .hasArgs()
                            .argName("STRING")
                            .desc("Specify HTTP multipart POST data")
                            .build());
            options.addOption(
                    Option.builder("G")
                            .longOpt("get")
                            .desc("Send the -d data with a HTTP GET")
                            .build());
            options.addOption(
                    Option.builder("H")
                            .longOpt("header")
                            .hasArgs()
                            .argName("LINE")
                            .desc("Pass custom header LINE to server")
                            .build());
            options.addOption(Option.builder("h").longOpt("help").desc("This help text").build());
            options.addOption(
                    Option.builder("i")
                            .longOpt("include")
                            .desc("Include protocol headers in the output")
                            .build());
            options.addOption(
                    Option.builder("k")
                            .longOpt("insecure")
                            .desc("Allow connections to SSL sites without certs")
                            .build());
            options.addOption(
                    Option.builder("o")
                            .longOpt("output")
                            .hasArg()
                            .argName("FILE")
                            .desc("Write to FILE instead of stdout")
                            .build());
            options.addOption(
                    Option.builder("e").longOpt("referer").hasArg().desc("Referer URL").build());
            options.addOption(
                    Option.builder("X")
                            .longOpt("request")
                            .hasArg()
                            .argName("COMMAND")
                            .desc("Specify request command to use")
                            .build());
            options.addOption(
                    Option.builder("T")
                            .longOpt("upload-file")
                            .hasArg()
                            .argName("FILE")
                            .desc("Transfer FILE to destination")
                            .build());
            options.addOption(
                    Option.builder("A")
                            .longOpt("user-agent")
                            .hasArg()
                            .argName("STRING")
                            .desc("Send User-Agent STRING to server")
                            .build());
            options.addOption(
                    Option.builder("H")
                            .longOpt("header")
                            .hasArgs()
                            .argName("LINE")
                            .desc("Pass custom header LINE to server")
                            .build());
            options.addOption(
                    Option.builder("v")
                            .longOpt("verbose")
                            .desc("Make the operation more talkative")
                            .build());
            options.addOption(
                    Option.builder("N")
                            .longOpt("repeat")
                            .hasArg()
                            .desc(" Number of requests to perform")
                            .build());
            options.addOption(
                    Option.builder("c")
                            .longOpt("clients")
                            .hasArg()
                            .desc(" Concurrent clients")
                            .build());
            return options;
        }

        public String getServiceName() {
            return serviceName;
        }

        public String getRegion() {
            return region;
        }

        public String getProfile() {
            return profile;
        }

        public int getConnectTimeout() {
            return connectTimeout;
        }

        public boolean isInclude() {
            return include;
        }

        public boolean isInsecure() {
            return insecure;
        }

        public OutputStream getOutput() throws IOException {
            if (output != null) {
                return new FileOutputStream(output);
            }
            if (nRequests > 1) {
                return new NullOutputStream();
            }
            return System.out;
        }

        public boolean isVerbose() {
            return verbose;
        }

        public int getNumberOfRequests() {
            return nRequests;
        }

        public int getClients() {
            return clients;
        }

        public String[] getForm() {
            return form;
        }

        public String[] getFormString() {
            return formString;
        }

        public String getRequestMethod() {
            if (forceGet) {
                return "GET";
            }
            return requestMethod == null ? "GET" : requestMethod;
        }

        public Map<String, String> getRequestHeaders() {
            Map<String, String> map = new HashMap<>();
            if (headers != null) {
                for (String header : headers) {
                    String[] pair = header.split(":", 2);
                    String key = pair[0].trim();
                    if ("content-type".equalsIgnoreCase(key)) {
                        key = "Content-Type";
                    } else if ("Referer".equalsIgnoreCase(key)) {
                        key = "Referer";
                    } else if ("User-Agent".equalsIgnoreCase(key)) {
                        key = "User-Agent";
                    } else if ("Content-Length".equalsIgnoreCase(key)) {
                        key = "Content-Length";
                    }
                    if (pair.length == 2) {
                        map.put(key, pair[1].trim());
                    }
                }
            }
            if (contentType != null) {
                map.put("Content-Type", contentType);
            }
            if (referer != null) {
                map.put("Referer", referer);
            }
            if (userAgent != null) {
                map.put("User-Agent", userAgent);
            }

            return map;
        }

        public String getUrl(String url) throws IOException {
            if (!forceGet) {
                return url;
            }

            if (data != null
                    || dataAscii != null
                    || dataBinary != null
                    || dataRaw != null
                    || dataUrlencode != null) {
                contentType = null;
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                addUrlEncodedData(bos, data, 1);
                addUrlEncodedData(bos, dataAscii, 1);
                addUrlEncodedData(bos, dataBinary, 1);
                addUrlEncodedData(bos, dataRaw, 2);
                addUrlEncodedData(bos, dataUrlencode, 3);
                bos.close();
                String queryString = bos.toString(StandardCharsets.UTF_8.name());
                int pos = url.indexOf('?');
                if (pos >= 0) {
                    if (pos == url.length() - 1) {
                        return url + queryString;
                    }
                    return url + '&' + queryString;
                }
                return url + '?' + queryString;
            }
            return url;
        }

        public byte[] getRequestBody() throws IOException {
            if (forceGet) {
                return null;
            }

            /*
             * Priority:
             *  1. --form, --form-string
             *  2. --data, --data-ascii, --data-binary, --data-raw, --data-urlencode
             *  3. --upload-file
             */
            if (form != null || formString != null) {
                requestMethod = requestMethod == null ? "POST" : requestMethod;
                MultipartEntityBuilder mb = MultipartEntityBuilder.create();
                addFormPart(mb, form, true);
                addFormPart(mb, formString, false);
                HttpEntity entity = mb.build();
                contentType = entity.getContentType().getValue();

                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                entity.writeTo(bos);
                return bos.toByteArray();
            }

            if (data != null
                    || dataAscii != null
                    || dataBinary != null
                    || dataRaw != null
                    || dataUrlencode != null) {
                requestMethod = requestMethod == null ? "POST" : requestMethod;
                contentType = ContentType.APPLICATION_FORM_URLENCODED.toString();
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                addUrlEncodedData(bos, data, 1);
                addUrlEncodedData(bos, dataAscii, 1);
                addUrlEncodedData(bos, dataBinary, 1);
                addUrlEncodedData(bos, dataRaw, 2);
                addUrlEncodedData(bos, dataUrlencode, 3);
                bos.close();
                return bos.toByteArray();
            }

            if (uploadFile != null) {
                requestMethod = requestMethod == null ? "PUT" : requestMethod;
                for (String header : headers) {
                    String[] pair = header.split(":", 2);
                    String key = pair[0].trim();
                    if ("content-type".equalsIgnoreCase(key)) {
                        contentType = pair[1].trim();
                    }
                }
                if (contentType == null) {
                    contentType = getMimeType(uploadFile).toString();
                }
                return readFile(uploadFile);
            }

            return null;
        }

        private byte[] readFile(String fileName) throws IOException {
            try (FileInputStream is = new FileInputStream(fileName)) {
                return IOUtils.toByteArray(is);
            }
        }

        private void addUrlEncodedData(ByteArrayOutputStream bos, String[] data, int encodeType)
                throws IOException {
            if (data == null) {
                return;
            }
            for (String content : data) {
                switch (encodeType) {
                    case 1: // data, data-ascii, data-binary
                        if (content.startsWith("@")) {
                            try (InputStream is = new FileInputStream(content.substring(1))) {
                                IOUtils.copy(is, bos);
                            }
                        } else {
                            bos.write(content.getBytes(StandardCharsets.UTF_8));
                        }
                        break;
                    case 2: // data-raw
                        bos.write(content.getBytes(StandardCharsets.UTF_8));
                        break;
                    case 3:
                    default:
                        writeUrlEncodedData(bos, content);
                        break;
                }
            }
        }

        private void writeUrlEncodedData(ByteArrayOutputStream bos, String content)
                throws IOException {
            String[] parameters = content.split("&");
            for (String parameter : parameters) {
                if (bos.size() > 0) {
                    bos.write('&');
                }

                String[] pair = parameter.split("=", 2);
                if (pair.length == 1) {
                    bos.write(readContentUrlEncoded(pair[0]));
                } else if (pair[0].isEmpty()) {
                    bos.write(readContentUrlEncoded(pair[1]));
                } else {
                    bos.write(pair[0].getBytes(StandardCharsets.UTF_8));
                    bos.write('=');
                    bos.write(readContentUrlEncoded(pair[1]));
                }
            }
        }

        private byte[] readContentUrlEncoded(String content) throws IOException {
            if (content.startsWith("@")) {
                File file = new File(content.substring(1));
                String value = IOUtils.toString(file.toURI(), StandardCharsets.UTF_8);
                return URLEncoder.encode(value, StandardCharsets.UTF_8.name())
                        .getBytes(StandardCharsets.UTF_8);
            }
            return URLEncoder.encode(content, StandardCharsets.UTF_8.name())
                    .getBytes(StandardCharsets.UTF_8);
        }

        private void addFormPart(MultipartEntityBuilder mb, String[] forms, boolean allowFile) {
            if (forms == null) {
                return;
            }

            for (String parameter : forms) {
                String[] tokens = parameter.split(";");
                String key = null;
                String value = null;
                String type = null;
                String fileName = null;
                for (String token : tokens) {
                    String[] pair = token.split("=", 2);
                    if ("type".equalsIgnoreCase(pair[0])) {
                        if (pair.length > 1) {
                            type = pair[1];
                        }
                    } else if ("filename".equalsIgnoreCase(pair[0])) {
                        if (pair.length > 1) {
                            fileName = pair[1];
                        }
                    } else {
                        key = pair[0];
                        value = pair.length > 1 ? pair[1] : "";
                    }
                }
                if (value == null) {
                    ContentType ct;
                    if (StringUtils.isEmpty(type)) {
                        ct = ContentType.TEXT_PLAIN;
                    } else {
                        ct = ContentType.create(type);
                    }
                    mb.addTextBody(key, "", ct);
                    return;
                }

                if (allowFile && value.startsWith("@")) {
                    File file = new File(value.substring(1));
                    if (StringUtils.isEmpty(fileName)) {
                        fileName = file.getName();
                    }
                    ContentType ct;
                    if (StringUtils.isEmpty(type)) {
                        ct = getMimeType(fileName);
                    } else {
                        ct = ContentType.create(type);
                    }
                    mb.addBinaryBody(key, file, ct, fileName);
                } else {
                    ContentType ct;
                    if (StringUtils.isEmpty(type)) {
                        ct = ContentType.TEXT_PLAIN;
                    } else {
                        ct = ContentType.create(type);
                    }
                    mb.addTextBody(key, value, ct);
                }
            }
        }

        private ContentType getMimeType(String fileName) {
            String ext = FilenameUtils.getExtension(fileName).toLowerCase(Locale.ENGLISH);
            switch (ext.toLowerCase(Locale.ENGLISH)) {
                case "txt":
                case "text":
                    return ContentType.TEXT_PLAIN;
                case "html":
                    return ContentType.TEXT_HTML;
                case "xhtml":
                    return ContentType.APPLICATION_XHTML_XML;
                case "xml":
                    return ContentType.APPLICATION_XML;
                case "json":
                    return ContentType.APPLICATION_JSON;
                case "jpg":
                case "jpeg":
                    return ContentType.IMAGE_JPEG;
                case "png":
                    return ContentType.IMAGE_PNG;
                case "gif":
                    return ContentType.IMAGE_GIF;
                case "bmp":
                    return ContentType.IMAGE_BMP;
                case "svg":
                    return ContentType.IMAGE_SVG;
                case "tiff":
                    return ContentType.IMAGE_TIFF;
                case "webp":
                    return ContentType.IMAGE_WEBP;
                default:
                    return ContentType.APPLICATION_OCTET_STREAM;
            }
        }
    }
}
