plugins {
    org.examples.javaProject
}

group = "org.example"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.itextpdf:itext-core:9.1.0")

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}
