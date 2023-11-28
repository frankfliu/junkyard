package com.amazonaws.awscurl;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

public final class JsonUtils {

    private static final Type MAP_TYPE = new TypeToken<Map<String, List<String>>>() {}.getType();

    static final Gson GSON = new Gson();

    private JsonUtils() {}

    static void getJsonList(JsonElement element, List<String> list, String name) {
        if (name == null) {
            name = "generated_text";
        }
        if (element.isJsonArray()) {
            JsonArray array = element.getAsJsonArray();
            for (int i = 0; i < array.size(); ++i) {
                getJsonList(array.get(i), list, name);
            }
        } else if (element.isJsonObject()) {
            JsonObject obj = element.getAsJsonObject();
            JsonElement e = obj.get(name);
            if (e != null) {
                if (e.isJsonPrimitive()) {
                    list.add(e.getAsString());
                } else if (e.isJsonArray()) {
                    JsonArray array = element.getAsJsonArray();
                    for (int i = 0; i < array.size(); ++i) {
                        JsonElement text = array.get(i);
                        if (text.isJsonPrimitive()) {
                            list.add(text.getAsString());
                        } else {
                            AwsCurl.logger.debug("Ignore element: {}", text);
                        }
                    }
                } else {
                    AwsCurl.logger.debug("Ignore element: {}", e);
                }
            }
        } else {
            AwsCurl.logger.debug("Ignore element: {}", element);
        }
    }

    @SuppressWarnings("PMD.SystemPrintln")
    static boolean processJsonLine(
            List<StringBuilder> list, long[] firstToken, OutputStream ps, String line, String name)
            throws IOException {
        if (name == null) {
            name = "outputs";
        }
        boolean hasError = false;
        boolean first = firstToken[0] == 0L;
        if (first) {
            firstToken[0] = System.nanoTime();
        }
        try {
            Map<String, List<String>> map = GSON.fromJson(line, MAP_TYPE);
            List<String> item = map.get(name);
            if (item != null) {
                if (list.isEmpty()) {
                    for (String s : item) {
                        list.add(new StringBuilder(s));
                    }
                } else {
                    for (int i = 0; i < item.size(); ++i) {
                        list.get(i).append(item.get(i));
                    }
                }
            }
        } catch (JsonParseException e) {
            if (first) {
                System.out.println("Invalid json line: " + line);
            }
            hasError = true;
        }

        ps.write(line.getBytes(StandardCharsets.UTF_8));
        ps.write(new byte[] {'\n'});
        return hasError;
    }
}
