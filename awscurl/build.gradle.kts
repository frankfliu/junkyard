plugins {
    org.examples.javaProject
}

version = "0.4.0"

dependencies {
    implementation(platform("ai.djl:bom:${libs.versions.djl.get()}"))
    implementation(libs.huggingface.tokenizers) {
        exclude(group = "org.apache.commons", module = "commons-compress")
    }
    implementation(libs.slf4j.simple)
    implementation(libs.commons.cli)
    implementation(libs.commons.codec)
    implementation(libs.apache.httpclient)
    implementation(libs.apache.httpmime)
    implementation("com.jayway.jsonpath:json-path:${libs.versions.jsonpath.get()}") {
        exclude(group = "net.minidev", module = "json-smart")
    }

    testImplementation(libs.netty.http)
    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}

tasks {
    jar {
        manifest {
            attributes["Main-Class"] = "com.amazonaws.awscurl.AwsCurl"
        }
        includeEmptyDirs = false
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE
        from(configurations.runtimeClasspath.get().map {
            if (it.isDirectory()) it else zipTree(it).matching {
                exclude("**/*.so")
                exclude("**/*.dylib")
                exclude("**/*.dll")
            }
        })

        doLast {
            providers.exec {
                workingDir = projectDir
                executable("sh")
                args(
                    "-c",
                    "cat src/main/scripts/stub.sh build/libs/awscurl*.jar > build/awscurl && chmod +x build/awscurl"
                )
            }
        }
    }
}

