plugins {
    org.examples.javaProject
}

version = "0.1.0"

dependencies {
    implementation(libs.commons.cli)
    implementation(libs.commons.codec)
    implementation(libs.commons.io)

    testImplementation(libs.testng)
}

tasks {
    jar {
        manifest {
            attributes["Main-Class"] = "com.amazonaws.jarjar.Bootstrap"
        }
    }
}
