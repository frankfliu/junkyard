plugins {
    org.examples.javaProject
}

version = "0.1.0"

dependencies {
    implementation(libs.commons.cli)
    implementation(libs.commons.codec)
    implementation(libs.commons.io)
    implementation(libs.apache.httpclient)
    implementation(libs.apache.httpmime)

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}

tasks {
    jar {
        manifest {
            attributes["Main-Class"] = "com.amazonaws.jarjar.Bootstrap"
        }
    }
}
