package com.amazonaws.awscurl;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class AWSCredentials {

    public static final String ACCESS_KEY_ENV_VAR = "AWS_ACCESS_KEY_ID";
    public static final String ALTERNATE_ACCESS_KEY_ENV_VAR = "AWS_ACCESS_KEY";
    public static final String SECRET_KEY_ENV_VAR = "AWS_SECRET_KEY";
    public static final String ALTERNATE_SECRET_KEY_ENV_VAR = "AWS_SECRET_ACCESS_KEY";
    public static final String AWS_SESSION_TOKEN_ENV_VAR = "AWS_SESSION_TOKEN";

    public static final String ACCESS_KEY_SYSTEM_PROPERTY = "aws.accessKeyId";
    public static final String SECRET_KEY_SYSTEM_PROPERTY = "aws.secretKey";
    public static final String SESSION_TOKEN_SYSTEM_PROPERTY = "aws.sessionToken";

    public static final String DEFAULT_PROFILE_NAME = "default";
    public static final String AWS_PROFILE_ENVIRONMENT_VARIABLE = "AWS_PROFILE";
    public static final String AWS_PROFILE_SYSTEM_PROPERTY = "aws.profile";

    private String awsAccessKey;
    private String awsSecretKey;
    private String sessionToken;
    private String region;

    public AWSCredentials(String awsAccessKey, String awsSecretKey, String sessionToken) {
        this(awsAccessKey, awsSecretKey, sessionToken, null);
    }

    public AWSCredentials(
            String awsAccessKey, String awsSecretKey, String sessionToken, String region) {
        this.awsAccessKey = awsAccessKey.trim();
        this.awsSecretKey = awsSecretKey.trim();
        if (sessionToken != null) {
            this.sessionToken = sessionToken.trim();
        }
        if (region != null) {
            this.region = region.trim();
        }
    }

    public String getAWSAccessKeyId() {
        return awsAccessKey;
    }

    public String getAWSSecretKey() {
        return awsSecretKey;
    }

    public String getSessionToken() {
        return sessionToken;
    }

    public String getRegion() {
        return region;
    }

    public static AWSCredentials getCredentials(String profile) {
        if (!StringUtils.isEmpty(profile)) {
            return loadFromProfile(profile);
        }

        String accessKey = System.getenv(ACCESS_KEY_ENV_VAR);
        if (accessKey == null) {
            accessKey = System.getenv(ALTERNATE_ACCESS_KEY_ENV_VAR);
        }
        String secretKey = System.getenv(SECRET_KEY_ENV_VAR);
        if (secretKey == null) {
            secretKey = System.getenv(ALTERNATE_SECRET_KEY_ENV_VAR);
        }
        String sessionToken = System.getenv(AWS_SESSION_TOKEN_ENV_VAR);

        if (!StringUtils.isEmpty(accessKey) && !StringUtils.isEmpty(secretKey)) {
            return new AWSCredentials(accessKey, secretKey, sessionToken);
        }

        accessKey = System.getProperty(ACCESS_KEY_SYSTEM_PROPERTY);
        secretKey = System.getProperty(SECRET_KEY_SYSTEM_PROPERTY);
        sessionToken = System.getProperty(SESSION_TOKEN_SYSTEM_PROPERTY);
        if (!StringUtils.isEmpty(accessKey) && !StringUtils.isEmpty(secretKey)) {
            return new AWSCredentials(accessKey, secretKey, sessionToken);
        }

        return loadFromProfile(getDefaultProfileName());
    }

    private static AWSCredentials loadFromProfile(String profile) {
        File home = new File(System.getProperty("user.home"), ".aws");
        if (!home.exists()) {
            return null;
        }
        File profileFile = new File(home, "credentials");
        if (profileFile.exists() && profileFile.isFile()) {
            return loadProfileCredentials(profileFile, profile);
        }

        profileFile = new File(home, "config");
        if (profileFile.exists() && profileFile.isFile()) {
            return loadProfileCredentials(profileFile, profile);
        }

        return null;
    }

    private static String getDefaultProfileName() {
        String profileName = System.getenv(AWS_PROFILE_ENVIRONMENT_VARIABLE);
        if (!StringUtils.isEmpty(profileName)) {
            return profileName;
        }

        profileName = System.getProperty(AWS_PROFILE_SYSTEM_PROPERTY);
        if (!StringUtils.isEmpty(profileName)) {
            return profileName;
        }

        return DEFAULT_PROFILE_NAME;
    }

    private static AWSCredentials loadProfileCredentials(File file, String profile) {
        Map<String, String> map = loadProfile(file, profile);
        String accessKey = map.get("aws_access_key_id");
        String secretKey = map.get("aws_secret_access_key");
        String sessionToken = map.get("aws_session_token");
        String region = map.get("region");
        if (!StringUtils.isEmpty(accessKey) && !StringUtils.isEmpty(secretKey)) {
            return new AWSCredentials(accessKey, secretKey, sessionToken, region);
        }
        return null;
    }

    private static Map<String, String> loadProfile(File file, String profile) {
        Map<String, String> map = new HashMap<>();
        try (Scanner scanner = new Scanner(file, StandardCharsets.UTF_8.name())) {
            boolean profileFound = false;
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }

                if (line.startsWith("[") && line.endsWith("]")) {
                    String profileName = line.substring(1, line.length() - 1);
                    if (profileName.startsWith("profile ")) {
                        profileName = profileName.substring("profile ".length()).trim();
                    }
                    if (profile.equalsIgnoreCase(profileName)) {
                        profileFound = true;
                    } else if (profileFound) {
                        return map;
                    }
                } else if (profileFound) {
                    String[] pair = line.split("=", 2);
                    if (pair.length != 2) {
                        throw new IllegalArgumentException(
                                "Invalid property format in the line: " + line);
                    }
                    map.put(pair[0].trim(), pair[1].trim());
                }
            }
        } catch (IOException e) {
            throw new AssertionError(e);
        }
        return map;
    }
}
