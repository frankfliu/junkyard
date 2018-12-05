package com.amazonaws.awscurl;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;

public class AWS4SignerRequestParams {

    private SignableRequest request;
    private long signingDateTimeMilli;
    private String scope;
    private String regionName;
    private String serviceName;
    private String formattedSigningDateTime;
    private String formattedSigningDate;
    private String signingAlgorithm;

    public AWS4SignerRequestParams(
            SignableRequest request,
            Date signingDateOverride,
            String regionNameOverride,
            String serviceName,
            String signingAlgorithm) {
        if (request == null) {
            throw new IllegalArgumentException("Request cannot be null");
        }
        if (signingAlgorithm == null) {
            throw new IllegalArgumentException("Signing Algorithm cannot be null");
        }

        this.request = request;
        this.signingDateTimeMilli =
                signingDateOverride != null
                        ? signingDateOverride.getTime()
                        : getSigningDate(request);

        TimeZone timeZone = TimeZone.getTimeZone("UTC");
        DateFormat dateFormat = new SimpleDateFormat("yyyyMMdd", Locale.ENGLISH);
        dateFormat.setTimeZone(timeZone);
        this.formattedSigningDate = dateFormat.format(new Date(signingDateTimeMilli));
        this.serviceName = serviceName;
        this.regionName = regionNameOverride;
        this.scope = generateScope(formattedSigningDate, serviceName, regionName);

        dateFormat = new SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.ENGLISH);
        dateFormat.setTimeZone(timeZone);
        this.formattedSigningDateTime = dateFormat.format(new Date(signingDateTimeMilli));
        this.signingAlgorithm = signingAlgorithm;
    }

    private long getSigningDate(SignableRequest request) {
        return System.currentTimeMillis() - request.getTimeOffset() * 1000L;
    }

    private String generateScope(String dateStamp, String serviceName, String regionName) {
        return dateStamp + "/" + regionName + "/" + serviceName + "/" + "aws4_request";
    }

    public SignableRequest getRequest() {
        return request;
    }

    public String getScope() {
        return scope;
    }

    public String getFormattedSigningDateTime() {
        return formattedSigningDateTime;
    }

    public long getSigningDateTimeMilli() {
        return signingDateTimeMilli;
    }

    public String getRegionName() {
        return regionName;
    }

    public String getServiceName() {
        return serviceName;
    }

    public String getFormattedSigningDate() {
        return formattedSigningDate;
    }

    public String getSigningAlgorithm() {
        return signingAlgorithm;
    }
}
