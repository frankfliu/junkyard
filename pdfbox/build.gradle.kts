plugins {
    org.examples.javaProject
}

group = "org.example"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.apache.pdfbox:pdfbox:3.0.5")
    implementation(libs.slf4j.simple)

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}
