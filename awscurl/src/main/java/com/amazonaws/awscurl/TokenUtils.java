package com.amazonaws.awscurl;

import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.training.util.DownloadUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class TokenUtils {

    private static final Logger logger = LoggerFactory.getLogger(TokenUtils.class);

    private static final HuggingFaceTokenizer tokenizer = getTokenizer();

    static void countTokens(
            List<? extends CharSequence> list, AtomicInteger tokens, SignableRequest request) {
        for (CharSequence item : list) {
            if (tokenizer != null) {
                Encoding encoding = tokenizer.encode(item.toString());
                tokens.addAndGet(encoding.getIds().length);
            } else {
                String[] token = item.toString().split("\\s");
                tokens.addAndGet(token.length);
            }
        }
        if (System.getenv("EXCLUDE_INPUT_TOKEN") != null) {
            tokens.addAndGet(-request.getInputTokens());
        }
    }

    private static HuggingFaceTokenizer getTokenizer() {
        try {
            Path cacheDir = Utils.getEngineCacheDir("tokenizers");
            Platform platform = Platform.detectPlatform("tokenizers");
            String classifier = platform.getClassifier();
            String version = platform.getVersion();
            Path dir = cacheDir.resolve(version + '-' + classifier);
            String libName = System.mapLibraryName("tokenizers");
            Path path = dir.resolve(libName);
            if (!Files.exists(path)) {
                Files.createDirectories(dir);
                String djlVersion = Engine.getDjlVersion().replaceAll("-SNAPSHOT", "");
                String url =
                        "https://publish.djl.ai/tokenizers/"
                                + version.split("-")[0]
                                + "/jnilib/"
                                + djlVersion
                                + '/'
                                + classifier
                                + '/'
                                + libName;
                DownloadUtils.download(new URL(url), path, null);
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to load HuggingFace tokenizer.", e);
        }

        HuggingFaceTokenizer.Builder builder = HuggingFaceTokenizer.builder();
        String name = System.getenv("TOKENIZER");
        if (name != null) {
            Path path = Paths.get(name);
            if (Files.exists(path)) {
                builder.optTokenizerPath(path);
            } else {
                builder.optTokenizerName(name);
            }
            try {
                return builder.build();
            } catch (Exception e) {
                logger.warn("", e);
                System.out.println(
                        "Invalid tokenizer: "
                                + name
                                + ", please unset environment variable TOKENIZER if don't want to"
                                + " use tokenizer");
            }
        }
        return null;
    }
}
