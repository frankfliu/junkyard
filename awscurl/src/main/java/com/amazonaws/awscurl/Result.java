package com.amazonaws.awscurl;

import ai.djl.util.Utils;

import java.io.IOException;
import java.io.Writer;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

class Result {

    private Boolean hasError;
    private String tokenizer;
    private double totalTimeMills;
    private int totalRequests;
    private int failedRequests;
    private double errorRate;
    private int concurrentClients;
    private Double tps;
    private Double tokenThroughput;
    private int totalTokens;
    private int tokenPerRequest;
    private Double averageLatency;
    private Double averageTokenLatency;
    private Double p50Latency;
    private Double p90Latency;
    private Double p99Latency;
    private Double timeToFirstByte;
    private Double p50TimeToFirstByte;
    private Double p90TimeToFirstByte;
    private Double p99TimeToFirstByte;

    Result() {
        tokenizer = Utils.getEnvOrSystemProperty("TOKENIZER");
    }

    public boolean hasError() {
        return hasError != null && hasError;
    }

    public void setHasError() {
        this.hasError = true;
    }

    public String getTokenizer() {
        return tokenizer;
    }

    public double getTotalTimeMills() {
        return totalTimeMills;
    }

    public void setTotalTimeMills(double totalTimeMills) {
        this.totalTimeMills = round(totalTimeMills);
    }

    public int getTotalRequests() {
        return totalRequests;
    }

    public void setTotalRequests(int totalRequests) {
        this.totalRequests = totalRequests;
    }

    public int getFailedRequests() {
        return failedRequests;
    }

    public void setFailedRequests(int failedRequests) {
        this.failedRequests = failedRequests;
    }

    public double getErrorRate() {
        return errorRate;
    }

    public void setErrorRate(double errorRate) {
        this.errorRate = round(errorRate);
    }

    public int getConcurrentClients() {
        return concurrentClients;
    }

    public void setConcurrentClients(int concurrentClients) {
        this.concurrentClients = concurrentClients;
    }

    public Double getTps() {
        return tps;
    }

    public void setTps(double tps) {
        this.tps = round(tps);
    }

    public Double getTokenThroughput() {
        return tokenThroughput;
    }

    public void setTokenThroughput(double tokenThroughput) {
        this.tokenThroughput = round(tokenThroughput);
    }

    public int getTotalTokens() {
        return totalTokens;
    }

    public void setTotalTokens(int totalTokens) {
        this.totalTokens = totalTokens;
    }

    public int getTokenPerRequest() {
        return tokenPerRequest;
    }

    public void setTokenPerRequest(int tokenPerRequest) {
        this.tokenPerRequest = tokenPerRequest;
    }

    public Double getAverageLatency() {
        return averageLatency;
    }

    public void setAverageLatency(double averageLatency) {
        this.averageLatency = round(averageLatency);
    }

    public Double getAverageTokenLatency() {
        return averageTokenLatency;
    }

    public void setAverageTokenLatency(Double averageTokenLatency) {
        this.averageTokenLatency = round(averageTokenLatency);
    }

    public Double getP50Latency() {
        return p50Latency;
    }

    public void setP50Latency(double p50Latency) {
        this.p50Latency = round(p50Latency);
    }

    public Double getP90Latency() {
        return p90Latency;
    }

    public void setP90Latency(double p90Latency) {
        this.p90Latency = round(p90Latency);
    }

    public Double getP99Latency() {
        return p99Latency;
    }

    public void setP99Latency(double p99Latency) {
        this.p99Latency = round(p99Latency);
    }

    public Double getTimeToFirstByte() {
        return timeToFirstByte;
    }

    public void setTimeToFirstByte(double timeToFirstByte) {
        this.timeToFirstByte = round(timeToFirstByte);
    }

    public Double getP50TimeToFirstByte() {
        return p50TimeToFirstByte;
    }

    public void setP50TimeToFirstByte(double p50TimeToFirstByte) {
        this.p50TimeToFirstByte = p50TimeToFirstByte;
    }

    public Double getP90TimeToFirstByte() {
        return p90TimeToFirstByte;
    }

    public void setP90TimeToFirstByte(double p90TimeToFirstByte) {
        this.p90TimeToFirstByte = p90TimeToFirstByte;
    }

    public Double getP99TimeToFirstByte() {
        return p99TimeToFirstByte;
    }

    public void setP99TimeToFirstByte(double p99TimeToFirstByte) {
        this.p99TimeToFirstByte = p99TimeToFirstByte;
    }

    @SuppressWarnings("PMD.SystemPrintln")
    public void print(boolean json, String path) throws IOException {
        if (json) {
            if (path == null) {
                System.out.println(JsonUtils.GSON_PRETTY.toJson(this));
            } else {
                Path filePath = Paths.get(path);
                try (Writer fbw = Files.newBufferedWriter(filePath)) {
                    fbw.write(JsonUtils.GSON_PRETTY.toJson(this));
                }
            }
            return;
        }

        System.out.printf("Total time: %.2f ms.%n", getTotalTimeMills());
        System.out.printf(
                "Non 200 responses: %d, error rate: %.2f%n", getFailedRequests(), getErrorRate());
        System.out.println("Concurrent clients: " + getConcurrentClients());
        System.out.println("Total requests: " + getTotalRequests());
        if (getTps() != null) {
            System.out.printf("TPS: %.2f/s%n", getTps());
            System.out.printf("Average Latency: %.2f ms.%n", getAverageLatency());
            System.out.printf("P50: %.2f ms.%n", getP50Latency());
            System.out.printf("P90: %.2f ms.%n", getP90Latency());
            System.out.printf("P99: %.2f ms.%n", getP99Latency());
        }
        if (getTotalTokens() > 0) {
            String n = getTokenizer() == null ? "word" : "token";
            System.out.printf("Total %s: %d%n", n, getTotalTokens());
            System.out.printf("%s/req: %d%n", n, getTokenPerRequest());
            System.out.printf("%s/s: %.2f/s%n", n, getTokenThroughput());
            if (getTokenizer() != null) {
                System.out.printf("Average Token Latency: %.2f ms.%n", getAverageTokenLatency());
            }
        }
        System.out.printf("Time to first byte: %.2f ms.%n", getTimeToFirstByte());
        System.out.printf("TTFB_P50: %.2f ms.%n", getP50TimeToFirstByte());
        System.out.printf("TTFB_P90: %.2f ms.%n", getP90TimeToFirstByte());
        System.out.printf("TTFB_P99: %.2f ms.%n", getP99TimeToFirstByte());
    }

    private static double round(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return value;
        }
        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(2, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
