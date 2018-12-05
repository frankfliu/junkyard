package com.amazonaws.awscurl;

public final class StringUtils {

    private StringUtils() {}

    public static boolean isEmpty(String str) {
        return str == null || str.isEmpty();
    }
}
