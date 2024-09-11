use std::fs;
use std::str::FromStr;

use reqwest::header::{HeaderMap, HeaderName, CONTENT_TYPE};
use reqwest::{Method, RequestBuilder};

use crate::args::Args;

pub(crate) struct Request {
    pub(crate) method: Method,
    pub(crate) url: String,
    pub(crate) headers: HeaderMap,
    body: Vec<u8>,
}

impl Request {
    pub fn new(args: &Args) -> Self {
        let method = match &args.request {
            None => {
                if args.data.is_none()
                    && args.data_raw.is_none()
                    && args.form.is_empty()
                    && args.form_string.is_empty()
                    && args.upload_file.is_none()
                    && args.dataset.is_none()
                {
                    Method::GET
                } else {
                    Method::POST
                }
            }
            Some(m) => Method::from_str(m.as_str()).unwrap(),
        };

        let mut url = args.url.clone();
        if method == Method::GET {
            if let Some(data) = &args.data {
                Self::append_url(&mut url, data);
            } else if let Some(data) = &args.data_raw {
                Self::append_url(&mut url, data);
            }
        };

        let mut content_type: Option<String> = None;
        let mut headers = HeaderMap::default();
        for line in args.header.clone() {
            let mut parts = line.splitn(2, ":");
            let first = parts.next().unwrap().trim();
            let second = parts.next().unwrap().trim();
            if "content-type".eq_ignore_ascii_case(first) {
                content_type = Some(second.into());
            }
            headers.append(
                HeaderName::from_str(first).unwrap(),
                second.parse().unwrap(),
            );
        }
        let mut body = Vec::default();
        if let Some(data) = &args.data {
            body.extend_from_slice(data.as_bytes());
            if content_type.is_none() && method == Method::POST {
                content_type = Some("application/x-www-form-urlencoded".into());
            }
        } else if let Some(data) = &args.data_raw {
            body.extend_from_slice(data.as_bytes());
            if content_type.is_none() && method == Method::POST {
                content_type = Some("application/x-www-form-urlencoded".into());
            }
        } else if !args.form.is_empty() || !args.form_string.is_empty() {
            // append_form_data(body, args.form, true);
            // append_form_data(body, args.form_string, true);
            if content_type.is_none() {
                content_type = Some("application/x-www-form-urlencoded".into());
            }
        } else if let Some(file) = &args.upload_file {
            if content_type.is_none() {
                let file_name = file.file_name().unwrap().to_str().unwrap();
                body.extend_from_slice(&fs::read(file).unwrap());
                content_type = Some(Self::get_mime_type(file_name).into());
            }
        }

        if let Some(t) = content_type {
            headers.insert(CONTENT_TYPE, t.parse().unwrap());
        }

        Self {
            method,
            url,
            headers,
            body,
        }
    }

    pub(crate) fn add_request_body(&self, builder: RequestBuilder) -> RequestBuilder {

    }

    pub(crate) fn get_request_body(&self) -> Vec<u8> {
        self.body.clone()
    }

    fn append_url(url: &mut String, query: &String) {
        match url.find('&') {
            None => {
                url.push('?');
            }
            Some(_) => {
                url.push('&');
            }
        };
        url.push_str(&query);
    }

    fn get_mime_type(file_name: &str) -> &str {
        match file_name {
            "txt" | "text" => "application/x-www-form-urlencoded",
            "json" => "application/json",
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "git" => "image/gif",
            "tiff" => "image/tiff",
            _ => "application/octet-stream",
        }
    }
}
